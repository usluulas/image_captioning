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
# Docker'da entrypoint.sh aynı değişkeni okuyup modeli indiriyor; ikisinin ayrışmaması için
# model adı ortam değişkeninden alınabiliyor. Docker dışında varsayılan aynen geçerli.
MODEL_ADI = os.environ.get("MODEL_ADI", "gemma3:4b-it-qat")
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
# NOT: Bu prompt oturum boyunca kural kural büyümüştü (847 kelimeye çıkmıştı) ve bir noktada
# 4GB VRAM'deki hesaplama/bellek baskısı üretimi yarıda kestiriyordu (anitkabir.jpg'de done=False).
# Aynı kuralları koruyup daha yoğun/derli toplu (~430 kelime) yeniden yazınca hem çökme ortadan
# kalktı hem de birkaç vaka (isim tanıma, OCR, renk) daha iyi çalıştı. Regresyon testi: kelebek,
# gitar, renk/eylem/bağlamsal sızma, OCR, oran/hava durumu, ikonik isim vakalarının hepsi korundu.
SISTEM_MESAJI = """
Sen profesyonel bir Arşiv Asistanısın. Sana verilen fotoğrafı betimleyeceksin.

KURAL 1 (DİL): Çıktındaki her kelime KESİNLİKLE TÜRKÇE olmalı — TEK İSTİSNA: görselde YAZILI bir
metin/başlık/logo görürsen bunu OLDUĞU GİBİ (orijinal dilinde) alıntıla, çevirme. Net okuyamadığın
yazıyı TAHMİN ETME, sadece "yazı var" de.

KURAL 2 (TARAMA): Görseli baştan sona tara — ön/arka plandaki TÜM nesneleri, kişileri/hayvanları,
GERÇEKTEN GÖRDÜĞÜN renkleri (birden fazla renk varsa hepsini say), konumları, eylemleri, ortamı,
yazıları — "gozlemler" alanına kısa maddeler halinde yaz.

KURAL 3 (ÖZEL İSİMLER): Bir nesnenin/canlının alt türünü, markasını/modelini ya da bir türün özel
adını (ör. bas mı elektro gitar mı, kelebek türü, "Telecaster") YALNIZCA somut kanıtın (tel sayısı,
ayırt edici desen, yazılı logo) varsa yaz; yoksa üst kategoriyi kullan (ör. "gitar", "kelebek").
İSTİSNA: DÜNYACA ÜNLÜ, kolay ayırt edilebilir landmark/anıt (Eyfel Kulesi, Anıtkabir, Big Ben,
Kolezyum), ünlü sanat eserleri (Van Gogh'un "Yıldızlı Gece"si) veya çok bilinen anime/film yapımları
(One Piece, Naruto, Spider-Man) için bu temkin GEÇERLİ DEĞİL — görsel tanıdıksa gerçek ismini KULLAN
(İngilizce olsa bile, çevirme).

KURAL 4 (UYDURMA): Görselde KESİNLİKLE olmayanı YAZMA. Emin olmadığın bir ayrıntıyı (tür, renk, ne
olduğu) ya yazma ya da somut görsel özellikleriyle (şekil, renk, malzeme) tarif et, bilinen bir
nesneye benzeterek TAHMİN ETME. Üç yaygın uydurma biçimine dikkat: (a) RENK/EYLEM SIZMASI — sahnede
baskın bir rengi/eylemi (bayrak kırmızısı, vb.) ilgisiz bir nesneye/kişiye yansıtma; (b) BAĞLAMSAL
SIZMA — bir karakter/nesne genelde başkalarıyla görülüyor diye (Iron Man→Avengers) o SPESİFİK
görselde olmayan başkalarını ekleme; (c) HAVA DURUMU/KONUM — gökyüzü rengi/bulut gibi somut kanıtın
ötesine geçip "yağmurlu", şehir/ülke ismi gibi doğrulanamaz genellemeler yapma.

KURAL 5 (CÜMLE): "gozlemler"e dayanarak akıcı TEK cümlelik betimleme üret; ana özneyi, eylemini, öne
çıkan detayları (renk, konum, ortam) içermeli — "Yeşil kanepenin üstünde yatan siyah bir kedi" gibi.
Sahnenin BASKIN konusuna odaklan; küçük/önemsiz bir detayı (sönük bir tabelanın kalıntı pikselleri
gibi) ana cümlenin odağı yapma, en fazla etikete ekle. Yoruma girme, "bu görselde şu var" deme,
doğrudan betimlemeyi yaz.

KURAL 6 (ETİKET): En fazla 6 TÜRKÇE etiket üret (nesneler, ortam, renk, eylem, tür). Ana cümleye
sığmayan önemli detayları etikete ekle. Kısa tut (1-3 kelime), tam cümle yazma, tekrar etme.

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


def _model_cagir(gorsel_verisi):
    return ollama.chat(
        model=MODEL_ADI,
        messages=[
            {'role': 'system', 'content': SISTEM_MESAJI},
            {
                'role': 'user',
                'content': 'Bu fotoğrafı Türkçe betimle ve etiketle.',
                'images': [gorsel_verisi],
            },
        ],
        format='json',
        options={**ORTAK_SECENEKLER, 'num_predict': NUM_PREDICT},
        keep_alive=0,  # Bu her fotoğraf için son çağrı:  bitince modeli bellekten hemen boşalt.
    )


def gorseli_betimle(dosya_yolu):
    gorsel_verisi = gorsel_veri_hazirla(dosya_yolu)
    response = _model_cagir(gorsel_verisi)

    # 4GB VRAM'de GPU zaman zaman üretimi ortada kesebiliyor (Ollama watchdog timeout);
    # bu durumda yanıt 'done=False' ve JSON yarım gelir. Bir kez otomatik yeniden dene.
    if not response.get('done'):
        response = _model_cagir(gorsel_verisi)

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

    # PowerShell, "&" gibi özel karakter içeren yolları kopyalarken "& '...'" biçiminde
    # sarar (call operator); gerçek yolu çıkarmak için önce bu sarmalamayı soyuyoruz.
    if dosya_yolu.startswith('& '):
        dosya_yolu = dosya_yolu[2:].strip()

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
