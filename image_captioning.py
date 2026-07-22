import os
import io
import json
import re    # Regex kütüphanesi (JSON Zırhı)
import ollama
from PIL import Image, ImageOps
import pillow_heif

# Ollama sunucusu HEIC/HEIF'i çözemiyor (yalnızca JPEG/PNG/WEBP gibi standart formatları
# destekliyor); bu yüzden iPhone fotoğrafları modele gitmeden önce PNG'ye çevrilir.
pillow_heif.register_heif_opener()


# ==========================================
# 0. MODEL VE OPTİMİZASYON AYARLARI
# ==========================================
MODEL_ADI = "gemma3:4b"

# Nihai sentez çağrısı hem görseli hem de NUM_TARAMA adet tarama metnini birden aldığı için
# tek-geçişli sürüme göre daha fazla bağlama ihtiyaç var; yine de 4GB VRAM'i zorlamamak için
# ölçülü tutuldu.
NUM_CTX = 6144

# Makine bu işlem sırasında başka iş için kullanılmıyor, tüm çekirdekler kullanılabilir.
NUM_THREAD = os.cpu_count() or 4

# Nihai çıktı (tek cümlelik betimleme + etiketler + tutarlılık notu) için yeterli pay.
NUM_PREDICT = 700

# --- SELF-CONSISTENCY (TUTARLILIK KONTROLÜ) AYARLARI ---
# Model tek bir üretimde kendinden emin ama YANLIŞ olabiliyor (ör. bir nesneyi kendinden emin
# şekilde "patates" sanması); "belirsizsen yazma" talimatı bunu yakalayamıyor çünkü model kendini
# belirsiz hissetmiyor. Çözüm: aynı görseli NUM_TARAMA kez bağımsız taratıp, taramalar arasındaki
# TUTARSIZLIĞI dışarıdan bir belirsizlik sinyali olarak kullanmak (nihai sentez adımında).
NUM_TARAMA = 3
TARAMA_TEMPERATURE = 0.4
TARAMA_NUM_PREDICT = 400

# temperature 0.0 (greedy) NİHAİ sentezi deterministik yapar. Tarama adımlarında ise kasıtlı
# olarak TARAMA_TEMPERATURE kullanılır (bkz. yukarısı) — taramaların birbirinden görece farklı
# çıkması, belirsiz noktaları ortaya çıkarmak için gerekli.
# repeat_penalty greedy/örneklemeli decoding'de oluşabilecek tekrar döngülerine karşı güvenlik payı.
ORTAK_SECENEKLER = {
    'num_ctx': NUM_CTX,
    'num_thread': NUM_THREAD,
    'temperature': 0.0,
    'repeat_penalty': 1.15,
}

DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")


# ==========================================
# 1. PROMPT ŞABLONLARI
# ==========================================
# NOT: Bu, TEK bir tarama çağrısının promptu. Kasıtlı olarak temkinsiz/serbest bırakıldı —
# NUM_TARAMA kez bağımsız çalıştırılıp modelin gerçekte NE KADAR emin olduğu, taramalar
# birbirine ne kadar UYDUĞUNA bakılarak (nihai sentez adımında) belirlenecek. Burada hedge
# ettirmeye çalışmak, her taramanın aynı temkinli/muğlak cevaba çekilmesine ve asıl belirsizlik
# sinyalinin kaybolmasına yol açar.
TARAMA_TALIMATI = """
Sen bir görsel analiz asistanısın. Sana verilen fotoğrafı en ince ayrıntısına kadar tara.
Ön plandaki ve arka plandaki TÜM nesneleri, kişileri/hayvanları, renklerini, konumlarını
(sağ/sol/üst/alt/ortada), eylemlerini, ortamı ve varsa görünen yazıları TÜRKÇE kısa maddeler
halinde yaz. Gördüğünü olduğu gibi, çekinmeden yaz; bu sadece bir ön tarama, temkinli davranmana
gerek yok. SADECE maddeleri yaz, başka açıklama/giriş cümlesi ekleme.
"""

# NOT: Bu, görsel + NUM_TARAMA adet bağımsız tarama metnini birlikte alan NİHAİ sentez çağrısının
# promptu. Asıl "belirsizse uydurma" mantığı burada uygulanıyor: model artık kendi iç belirsizliğini
# tahmin etmek zorunda değil, taramalar arasındaki somut ANLAŞMAZLIĞA bakabiliyor.
SISTEM_MESAJI = """
Sen profesyonel bir Arşiv Asistanısın. Sana bir fotoğraf ve o fotoğrafın birbirinden bağımsız
yapılmış {NUM_TARAMA} ayrı taramasını vereceğim. Görevin bunlara dayanarak nihai TÜRKÇE betimleme
ve etiketleri üretmek.
KURAL 1: Çıktındaki her bir kelime KESİNLİKLE TÜRKÇE olmalıdır.
KURAL 2: Taramaların HEPSİNDE (ya da açık çoğunluğunda) TUTARLI şekilde geçen nesne, renk, konum
ve detayları gerçek kabul et; bunları "tutarli_detaylar" alanına kısa maddeler halinde yaz.
KURAL 3: Bir tarama diğerlerinde hiç geçmeyen ya da diğerleriyle ÇELİŞEN bir nesne/detaydan
bahsediyorsa (ör. 3 taramadan sadece 1'i bir nesneyi "patates" derken diğerleri hiç bahsetmiyor
ya da farklı bir şey söylüyorsa), bunu KESİN bilgi sayma. Böyle bir anlaşmazlık durumunda o
nesneyi/detayı ya atla ya da sadece gördüğün somut görsel özellikleriyle (şekil, renk, malzeme
görünümü) genel bir ifadeyle geç — ona benzeyen bilinen bir nesnenin adını TAHMİN ETME.
KURAL 4: Renkleri taramalarda GERÇEKTEN belirtildiği gibi yaz (bir hayvanın/nesnenin üzerinde
birden fazla renk lekesi varsa hepsini say, tek bir genel renge indirgeme). Bir nesnenin/canlının
alt türünü ya da modelini (ör. bas gitar mı elektro gitar mı, kedi ırkı ne) sadece taramaların
net ve tutarlı şekilde desteklediği durumda belirt; taramalar arasında bu konuda çelişki veya
belirsizlik varsa üst kategoriyi kullan (ör. sadece "gitar", "kedi").
KURAL 5: "tutarli_detaylar" listene dayanarak akıcı TEK bir cümlelik betimleme üret; ana özneyi,
durumunu/eylemini ve öne çıkan görsel detayları (renk, konum, ortam) içermeli. Üslup örneği:
"Yeşil kanepenin üstünde yatan siyah bir kedi". Tutarlı olup ana cümleye sığmayan ikincil
detayları KAYBETME, KURAL 7 uyarınca etiketlere yansıt.
KURAL 6: Yoruma girme, "bu görselde şu var" gibi meta cümleler kurma; "aciklama" alanına DOĞRUDAN
betimlemeyi yaz.
KURAL 7: Fotoğrafı en iyi yansıtan en fazla 6 adet TÜRKÇE etiket üret (nesneler, ortam, renk,
eylem, tür/kategori gibi), sadece "tutarli_detaylar" listenden. Her etiket kısa olmalı (tek kelime
veya en fazla 2-3 kelimelik öbek), tam cümle YAZMA.
KURAL 8: Aynı anlama gelen etiketi tekrar etme.

Yanıtını KESİNLİKLE SADECE aşağıdaki JSON formatında ver:
{{
    "tutarli_detaylar": "Taramalar arasında tutarlı olan nesne, renk, konum ve eylemlerin kısa maddeler halindeki listesi",
    "aciklama": "Görselin KESİNLİKLE TÜRKÇE, tek cümlelik betimlemesi",
    "etiketler": ["etiket1", "etiket2", "etiket3", "etiket4", "etiket5", "etiket6"]
}}
"""


# ==========================================
# 2. YARDIMCI FONKSİYON: JSON YAKALAYICI (REGEX YEDEKLİ)
# ==========================================
def guvenli_json_ayikla(ham_metin):
    try:
        return json.loads(ham_metin.strip())
    except json.JSONDecodeError:
        # format='json' zaten geçerli JSON garanti eder; bu regex sadece ekstra güvenlik katmanı.
        match = re.search(r'\{.*\}', ham_metin, re.DOTALL)
        if not match:
            raise ValueError("Yapay zeka yanıtı içinde geçerli bir JSON bloğu bulunamadı.")
        return json.loads(match.group(0))


def etiketleri_normallestir(etiketler):
    if not isinstance(etiketler, list):
        return []
    gorulen = set()
    sonuc = []
    for etiket in etiketler:
        temiz = str(etiket).strip()
        anahtar = temiz.lower()
        if temiz and anahtar not in gorulen:
            gorulen.add(anahtar)
            sonuc.append(temiz)
    return sonuc


# ==========================================
# 3. GÖRSEL ANALİZ
# ==========================================
def gorsel_veri_hazirla(dosya_yolu):
    # Ollama'ya dosya yolu doğrudan verilirse ham piksel verisi hiçbir düzeltme yapılmadan
    # gönderilir. Telefon fotoğraflarının çoğu EXIF "Orientation" etiketiyle döndürülerek
    # gösterilir (ör. sensör yatay kaydeder, etiket 90° çevirmeyi işaret eder); bu etiket
    # uygulanmazsa model görseli yan yatmış haliyle "görür" ve tamamen alakasız bir betimleme
    # üretebilir. Bu yüzden TÜM görseller PIL ile açılıp EXIF'e göre düzeltiliyor ve PNG
    # baytları olarak gönderiliyor (HEIC/HEIF için PNG'ye çevirmek zaten zorunluydu).
    with Image.open(dosya_yolu) as img:
        img = ImageOps.exif_transpose(img)
        arabellek = io.BytesIO()
        img.convert("RGB").save(arabellek, format="PNG")
        return arabellek.getvalue()


def gorseli_tara(gorsel_verisi):
    yanit = ollama.chat(
        model=MODEL_ADI,
        messages=[
            {'role': 'system', 'content': TARAMA_TALIMATI},
            {'role': 'user', 'content': 'Bu fotoğrafı tara.', 'images': [gorsel_verisi]},
        ],
        options={
            'num_ctx': NUM_CTX,
            'num_thread': NUM_THREAD,
            'temperature': TARAMA_TEMPERATURE,
            'repeat_penalty': 1.15,
            'num_predict': TARAMA_NUM_PREDICT,
        },
        keep_alive='5m',  # Ardışık taramalar arasında modeli bellekte tut, her seferinde yeniden yükleme.
    )
    return yanit['message']['content'].strip()


def gorseli_betimle(dosya_yolu):
    gorsel_verisi = gorsel_veri_hazirla(dosya_yolu)

    taramalar = [gorseli_tara(gorsel_verisi) for _ in range(NUM_TARAMA)]
    taramalar_metni = "\n\n".join(
        f"--- TARAMA {i + 1} ---\n{tarama}" for i, tarama in enumerate(taramalar)
    )

    response = ollama.chat(
        model=MODEL_ADI,
        messages=[
            {'role': 'system', 'content': SISTEM_MESAJI.format(NUM_TARAMA=NUM_TARAMA)},
            {
                'role': 'user',
                'content': f"Aşağıda bu fotoğrafın {NUM_TARAMA} bağımsız taraması var:\n\n{taramalar_metni}\n\nBunlara dayanarak nihai analizi JSON olarak üret.",
                'images': [gorsel_verisi],
            },
        ],
        format='json',
        options={**ORTAK_SECENEKLER, 'num_predict': NUM_PREDICT},
        keep_alive=0,  # Bu her fotoğraf için son çağrı: bitince modeli bellekten hemen boşalt.
    )

    try:
        data = guvenli_json_ayikla(response['message']['content'])
        aciklama = data.get("aciklama", "Betimleme çıkarılamadı.")
        etiketler = etiketleri_normallestir(data.get("etiketler", []))
        return f"BETİMLEME:\n{aciklama}\n\nETİKETLER:\n{', '.join(etiketler)}"
    except Exception as e:
        return f"[HATA] Yapay zeka istenen formatta yanıt veremedi (Sebep: {str(e)[:50]}).\n\nHam Yanıt:\n" + response['message']['content']


# ==========================================
# 4. ANA TERMİNAL İŞLEYİŞİ
# ==========================================
def main():
    print("=" * 60)
    print(" PROFESYONEL AI GÖRSEL BETİMLEYİCİ (JSON DESTEKLİ) ")
    print("=" * 60)
    print(f"[Model] {MODEL_ADI}, num_ctx: {NUM_CTX}, num_thread: {NUM_THREAD}")

    dosya_yolu = input("Lütfen betimlenecek fotoğrafın yolunu girin:\n> ").strip()

    if dosya_yolu.startswith('"') and dosya_yolu.endswith('"'):
        dosya_yolu = dosya_yolu[1:-1]
    if dosya_yolu.startswith("'") and dosya_yolu.endswith("'"):
        dosya_yolu = dosya_yolu[1:-1]

    if not os.path.exists(dosya_yolu):
        print("\nHata: Belirtilen dosya bulunamadı! Yolu doğru kopyaladığınızdan emin olun.")
        return

    _, uzanti = os.path.splitext(dosya_yolu)
    uzanti = uzanti.lower()

    if uzanti not in DESTEKLENEN_UZANTILAR:
        print(f"\nHata: Sadece {', '.join(DESTEKLENEN_UZANTILAR)} formatları desteklenmektedir.")
        return

    try:
        print("\n[İşlem] Fotoğraf analiz ediliyor, lütfen bekleyin...")

        ai_yaniti = gorseli_betimle(dosya_yolu)

        print("\n" + "=" * 60)
        print(f"--- DOSYA: {os.path.basename(dosya_yolu)} ---")
        print("=" * 60)
        print(ai_yaniti)
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[HATA] Bir hata oluştu: {str(e)}")
        print(f"Lütfen arka planda Ollama'nın çalıştığından ('ollama run {MODEL_ADI}') emin olun.")


if __name__ == "__main__":
    main()
