import os
import io
import json
import re    # Regex kütüphanesi (JSON Zırhı)
import ollama
from PIL import Image
import pillow_heif

# Ollama sunucusu HEIC/HEIF'i çözemiyor (yalnızca JPEG/PNG/WEBP gibi standart formatları
# destekliyor); bu yüzden iPhone fotoğrafları modele gitmeden önce PNG'ye çevrilir.
pillow_heif.register_heif_opener()


# ==========================================
# 0. MODEL VE OPTİMİZASYON AYARLARI
# ==========================================
MODEL_ADI = "gemma3:4b"

# Görsel + kısa metin promptu için 8192'ye gerek yok; daha düşük context KV cache'i
# küçültüp 4GB VRAM'de görsel kodlayıcıya daha fazla bellek bırakır.
NUM_CTX = 4096

# Makine bu işlem sırasında başka iş için kullanılmıyor, tüm çekirdekler kullanılabilir.
NUM_THREAD = os.cpu_count() or 4

# Çıktı zaten kısa bir JSON (tek cümlelik betimleme + etiketler) olacağı için sınırlandı.
NUM_PREDICT = 400

# temperature 0.0 (greedy) modeli deterministik yapar, halüsinasyonu azaltır.
# repeat_penalty greedy decoding'de oluşabilecek tekrar döngülerine karşı güvenlik payı.
ORTAK_SECENEKLER = {
    'num_ctx': NUM_CTX,
    'num_thread': NUM_THREAD,
    'temperature': 0.0,
    'repeat_penalty': 1.15,
}

DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")
HEIC_UZANTILARI = (".heic", ".heif")


# ==========================================
# 1. PROMPT ŞABLONU
# ==========================================
SISTEM_MESAJI = """
Sen profesyonel bir Arşiv Asistanısın. Sana verilen fotoğrafı betimleyeceksin.
KURAL 1: Çıktındaki her bir kelime KESİNLİKLE TÜRKÇE olmalıdır.
KURAL 2: Betimleme akıcı, tek bir cümle olmalı; görseldeki ana özneyi, durumunu/eylemini ve öne
çıkan görsel detayları (renk, konum, ortam) içermeli. Üslup örneği: "Yeşil kanepenin üstünde
yatan siyah bir kedi".
KURAL 3: Görselde KESİNLİKLE olmayan hiçbir şeyi UYDURMA. Bir ayrıntıdan emin değilsen yazma.
KURAL 4: Yoruma girme, "bu görselde şu var" gibi meta cümleler kurma; DOĞRUDAN betimlemeyi yaz.
KURAL 5: Fotoğrafı en iyi yansıtan en fazla 6 adet TÜRKÇE etiket üret (nesneler, ortam, renk,
eylem, tür/kategori gibi). Her etiket kısa olmalı (tek kelime veya en fazla 2-3 kelimelik öbek),
tam cümle YAZMA.
KURAL 6: Aynı anlama gelen etiketi tekrar etme.

Yanıtını KESİNLİKLE SADECE aşağıdaki JSON formatında ver:
{
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
    # Ollama'ya HEIC/HEIF gönderilemez; PNG'ye çevirip baytları döndürüyoruz.
    # Diğer formatlarda dosya yolu doğrudan geçilir (gereksiz yeniden kodlamayı önler).
    _, uzanti = os.path.splitext(dosya_yolu)
    if uzanti.lower() not in HEIC_UZANTILARI:
        return dosya_yolu

    with Image.open(dosya_yolu) as img:
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
        keep_alive=0,  # Bu her fotoğraf için son çağrı: bitince modeli bellekten hemen boşalt
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
