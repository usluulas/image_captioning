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
MODEL_ADI = "gemma3:4b-it-qat"
# QAT (quantization-aware training) sürümü, standart Q4_K_M sürümüne göre gözle görülür daha
# isabetli sonuç veriyor (ör. hoparlörü "sehpa" sanmıyor, haber görsellerinde bağlamı daha doğru
# yakalıyor) — bedeli ~1.5-2.5x daha yavaş çalışması (model 4.0GB, 4GB VRAM'e daha sıkışık sığıyor).
# Bilinçli bir kalite/hız tercihi.

# Görsel + kısa metin promptu için 8192'ye gerek yok; daha düşük context KV cache'i
# küçültüp 4GB VRAM'de görsel kodlayıcıya daha fazla bellek bırakır.
NUM_CTX = 4096

# Makine bu işlem sırasında başka iş için kullanılmıyor, tüm çekirdekler kullanılabilir.
NUM_THREAD = os.cpu_count() or 4

# Modelin önce görseli ayrıntılı taraması ("gozlemler" alanı), sonra betimleme + etiketleri
# üretmesi gerektiği için tek cümlelik çıktıya göre daha geniş bir pay bırakıldı.
NUM_PREDICT = 700

# temperature 0.0 (greedy) modeli deterministik yapar, halüsinasyonu azaltır.
# repeat_penalty greedy decoding'de oluşabilecek tekrar döngülerine karşı güvenlik payı.
ORTAK_SECENEKLER = {
    'num_ctx': NUM_CTX,
    'num_thread': NUM_THREAD,
    'temperature': 0.0,
    'repeat_penalty': 1.15,
}

DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")


# ==========================================
# 1. PROMPT ŞABLONU
# ==========================================
SISTEM_MESAJI = """
Sen profesyonel bir Arşiv Asistanısın. Sana verilen fotoğrafı betimleyeceksin.
KURAL 1: Çıktındaki her bir kelime KESİNLİKLE TÜRKÇE olmalıdır.
KURAL 2: Önce görseli baştan sona TARA: ön plandaki ve arka plandaki TÜM nesneleri, kişileri/
hayvanları, renklerini, konumlarını (sağ/sol/üst/alt/ortada), eylemlerini, ortamı ve varsa görünen
yazıları tek tek fark et; bunları "gozlemler" alanına kısa maddeler halinde yaz. Renkleri GERÇEKTEN
GÖRDÜĞÜN gibi yaz (bir hayvanın/nesnenin üzerinde birden fazla renk lekesi varsa hepsini say, tek
bir genel renge indirgeme). Bir nesnenin/canlının alt türünü ya da modelini (ör. bas gitar mı elektro
gitar mı, kedi ırkı ne) belirtmeden önce bunu destekleyen somut, sayılabilir bir kanıtı (tel sayısı,
gövde/kafa şekli, ayırt edici desen) gözleminde belirt; böyle bir kanıtın yoksa daha genel ve güvenli
bir terim kullan (ör. "gitar", "kedi"). Görsel olarak birbirine çok benzeyen alt türler arasında (ör.
bas gitar / elektro gitar / akustik gitar; kedi / kaplan; farklı köpek ırkları) KESİN ayrım
yapamıyorsan asla spesifik olanı seçme, üst kategoriyi (ör. sadece "gitar") kullan.
KURAL 3: "gozlemler" taramana dayanarak akıcı TEK bir cümlelik betimleme üret; ana özneyi,
durumunu/eylemini ve öne çıkan görsel detayları (renk, konum, ortam) içermeli. Üslup örneği:
"Yeşil kanepenin üstünde yatan siyah bir kedi". Taramada fark edip ana cümleye sığdıramadığın
ikincil detayları KAYBETME, KURAL 6 uyarınca etiketlere yansıt.
KURAL 4: Görselde KESİNLİKLE olmayan hiçbir şeyi UYDURMA. Bir ayrıntıdan (tür, model, tam renk dahil)
emin değilsen o ayrıntıyı YAZMA veya daha genel bir terimle belirt; net kanıtın olmadığı spesifik bir
iddiada bulunma. Bu kural bir nesnenin ALT TÜRÜ için olduğu kadar TEMEL KİMLİĞİ için de geçerlidir:
bir nesnenin ne olduğundan (ör. yiyecek mi, oyuncak mı, alet mi) emin değilsen, ona benzeyen bilinen
bir nesnenin adını TAHMİN ETME; bunun yerine sadece gördüğün somut görsel özellikleriyle tarif et
(şekil, renk, malzeme görünümü — ör. "yuvarlak, cilalı, açık kahverengi bir nesne").
KURAL 5: Yoruma girme, "bu görselde şu var" gibi meta cümleler kurma; "aciklama" alanına DOĞRUDAN
betimlemeyi yaz.
KURAL 6: Fotoğrafı en iyi yansıtan en fazla 6 adet TÜRKÇE etiket üret (nesneler, ortam, renk,
eylem, tür/kategori gibi). Taramada fark edip ana cümleye giremeyen önemli nesne/detayları
öncelikle etiketlere ekle. Her etiket kısa olmalı (tek kelime veya en fazla 2-3 kelimelik öbek),
tam cümle YAZMA.
KURAL 7: Aynı anlama gelen etiketi tekrar etme.

Yanıtını KESİNLİKLE SADECE aşağıdaki JSON formatında ver:
{
    "gozlemler": "Görseldeki tüm nesne, renk, konum ve eylemlerin kısa maddeler halinde taraması",
    "aciklama": "Görselin KESİNLİKLE TÜRKÇE, tek cümlelik betimlemesi",
    "etiketler": ["etiket1", "etiket2", "etiket3", "etiket4", "etiket5", "etiket6"]
}
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


def gorseli_betimle(dosya_yolu):
    response = ollama.chat(
        model=MODEL_ADI,
        messages=[
            {'role': 'system', 'content': SISTEM_MESAJI},
            {
                'role': 'user',
                'content': 'Bu fotoğrafı Türkçe betimle ve etiketle.',
                'images': [gorsel_veri_hazirla(dosya_yolu)],
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
