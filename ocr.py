"""Görseldeki yazıyı Tesseract ile okur.

Neden ayrı bir motor: gemma3 modeli afiş/logo gibi stilize yazıları iyi okuyor ama yoğun
düz metinde (haber görselindeki 9 satırlık fiyat tablosu gibi) tamamen başarısız oluyordu ve
promptla çözmeye çalıştığımız iki turda uydurma metin üretmeye başladı (yazısız bir manzara
fotoğrafına "Lago di Braies", voleybol fotoğrafına "Orhan Kemal" yazdı). Tesseract okuyamadığında
boş döner, uydurmaz — arşiv çıktısı için belirleyici olan bu.

Ölçüm (2026-07-29, 9 test görseli): fiyat tablosunda 9 fiyatın 8'i ve manşet birebir doğru;
yazısız dört görselin dördünde de boş döndü. Buna karşılık stilize afiş başlıklarını
("LOST", "SPIDER-MAN", "Fender") okuyamıyor — onlar zaten modelin betimlemesine giriyor.
"""

import os
import re
import shutil

from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - HEIC desteği yoksa OCR yine de çalışsın
    pass


# ==========================================
# 0. AYARLAR
# ==========================================
# Tesseract bir sistem programı; Windows'ta ve Docker'da farklı yerde kurulu oluyor.
# MODEL_ADI'ndaki gibi ortam değişkeninden geçersiz kılınabiliyor.
VARSAYILAN_YOLLAR = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
)

# Dil dosyalarının bulunduğu dizin. Windows'ta winget kurulumu yalnızca İngilizce getirdiği ve
# Program Files'a yazmak yönetici izni istediği için tur.traineddata kullanıcı dizinine kuruldu.
# Docker'da apt-get standart yere kurar; orada bu değişkene gerek kalmaz.
TESSDATA_DIZINI = os.environ.get(
    "TESSDATA_DIZINI", os.path.join(os.environ.get("LOCALAPPDATA", ""), "tessdata")
)

DILLER = os.environ.get("OCR_DILLERI", "tur+eng")

# Tesseract'ın kelime başına verdiği güven yüzdesi. 60 ölçümle seçildi: 75'e çıkarınca gerçek
# metin parçalanıyordu ("SON DAKİKA" -> "SON"), daha aşağısında çöp satırlar geçiyordu.
GUVEN_ESIGI = float(os.environ.get("OCR_GUVEN_ESIGI", "60"))

# Büyük telefon fotoğraflarında OCR 17 saniyeye kadar çıkıyordu; küçültünce saniyelere iniyor ve
# okuma kalitesi düşmüyor (Tesseract zaten ~300 DPI eşdeğerinden fazlasından faydalanmıyor).
AZAMI_KENAR = 2500

# Küçük görselleri büyütmek Tesseract'ın doğruluğunu belirgin artırıyor (haber görselinde manşet
# ham halde "EMEKLİ a O's IZ ARAÇ LİSTESİ Bi" çıkarken büyütülünce birebir doğru okundu).
ASGARI_KENAR = 1600


def _tesseract_bul():
    """Kurulu tesseract çalıştırılabilirini bul; yoksa None dön."""
    elle = os.environ.get("TESSERACT_YOLU")
    if elle and os.path.isfile(elle):
        return elle
    yol = shutil.which("tesseract")
    if yol:
        return yol
    for aday in VARSAYILAN_YOLLAR:
        if os.path.isfile(aday):
            return aday
    return None


# ==========================================
# 1. GÖRSEL HAZIRLAMA
# ==========================================
def _kucult(img):
    if max(img.size) > AZAMI_KENAR:
        oran = AZAMI_KENAR / max(img.size)
        img = img.resize((int(img.width * oran), int(img.height * oran)), Image.LANCZOS)
    return img


def _varyant(img, mod):
    """Tek bir ön işleme yeterli olmuyor: sepya tonlu afişte ham görüntü, koyu zeminli yoğun
    metinde gri tonlama + büyütme kazanıyor. İkisi de denenip skoru yüksek olan seçiliyor."""
    if mod == "ham":
        return img.copy()
    gri = ImageOps.grayscale(img)
    if max(gri.size) < ASGARI_KENAR:
        oran = ASGARI_KENAR / max(gri.size)
        gri = gri.resize((int(gri.width * oran), int(gri.height * oran)), Image.LANCZOS)
    return gri


# ==========================================
# 2. SATIR ÇIKARMA VE TEMİZLİK
# ==========================================
def _cop_mu(metin):
    """Anlamsız OCR artıklarını ("#)", "ar Te", "»*)") ele."""
    if len(metin) < 3:
        return True
    alfanumerik = sum(c.isalnum() for c in metin)
    if alfanumerik < 2:
        return True
    # "Ml Al" gibi, tamamı 1-2 harflik parçalardan oluşan satırlar gerçek metin değil; bunlar
    # afiş dokusundan/gölgesinden çıkan artıklar. Tek kelimelik kısa yazılar ("BIG") korunur.
    if max((len(p) for p in metin.split()), default=0) < 3:
        return True
    return alfanumerik / len(metin) < 0.55


def _bastaki_copu_kirp(metin):
    # Afişlerdeki raptiye/madde imi ikonları "#)", "(60", "@" gibi artıklara dönüşüyor.
    # Yalnızca BAŞTAKİ semboller kırpılır; sondakiler korunur, yoksa "BELLİ OLDU!" ünlemini
    # kaybederiz.
    return re.sub(r"^[^0-9A-Za-zÇĞİÖŞÜçğıöşü]+", "", metin).strip()


def _satirlari_cikar(img, pytesseract, yapilandirma):
    """image_to_data çıktısını satırlara topla; düşük güvenli kelimeleri at.

    Her satır için dikey konum (top) da tutulur, çıktı görseldeki sırayla verilebilsin diye.
    """
    veri = pytesseract.image_to_data(
        img, lang=DILLER, config=yapilandirma, output_type=pytesseract.Output.DICT
    )

    gruplar = {}
    for i, kelime in enumerate(veri["text"]):
        temiz = (kelime or "").strip()
        if not temiz:
            continue
        try:
            guven = float(veri["conf"][i])
        except (TypeError, ValueError):
            continue
        if guven < GUVEN_ESIGI:
            continue
        anahtar = (veri["block_num"][i], veri["par_num"][i], veri["line_num"][i])
        gruplar.setdefault(anahtar, []).append((temiz, guven, veri["top"][i]))

    satirlar = []
    for kelimeler in gruplar.values():
        metin = _bastaki_copu_kirp(" ".join(k for k, _, _ in kelimeler))
        if _cop_mu(metin):
            continue
        ortalama = sum(g for _, g, _ in kelimeler) / len(kelimeler)
        ust = min(t for _, _, t in kelimeler)
        satirlar.append((metin, ortalama, ust))
    return satirlar


def _skor(satirlar):
    """Varyant kalitesi: okunan karakter miktarı x ortalama güven."""
    if not satirlar:
        return 0.0
    karakter = sum(len(m) for m, _, _ in satirlar)
    ortalama = sum(g for _, g, _ in satirlar) / len(satirlar)
    return karakter * ortalama


def _alt_dizeleri_ele(satirlar):
    """Bir satır diğerinin içinde geçiyorsa kısa olanı at ("SON" vs "SON DAKİKA")."""
    uzundan_kisaya = sorted(satirlar, key=lambda s: len(s[0]), reverse=True)
    tutulan = []
    for metin, guven, ust in uzundan_kisaya:
        normal = " ".join(metin.lower().split())
        if any(normal in " ".join(t.lower().split()) for t, _, _ in tutulan):
            continue
        tutulan.append((metin, guven, ust))
    return tutulan


# ==========================================
# 3. DIŞA AÇIK FONKSİYON
# ==========================================
def yazilari_oku(dosya_yolu):
    """Görseldeki okunabilir yazıları, görseldeki sırayla liste olarak döndürür.

    Tesseract kurulu değilse ya da herhangi bir hata olursa BOŞ LİSTE döner — OCR bu programda
    ek bir özellik, betimlemeyi engellememeli.
    """
    tesseract_yolu = _tesseract_bul()
    if not tesseract_yolu:
        return []

    try:
        import pytesseract
    except ImportError:
        return []

    pytesseract.pytesseract.tesseract_cmd = tesseract_yolu

    # TESSDATA_PREFIX ortam değişkeni alt sürece güvenilir geçmiyor; --tessdata-dir bayrağı
    # kesin sonuç veriyor. Yolda boşluk varsa pytesseract config'i boşluktan böldüğü için
    # bayrak kullanılamaz; o durumda Tesseract kendi varsayılan dizinine düşer.
    yapilandirma = ""
    if TESSDATA_DIZINI and os.path.isdir(TESSDATA_DIZINI) and " " not in TESSDATA_DIZINI:
        yapilandirma = f"--tessdata-dir {TESSDATA_DIZINI}"

    try:
        with Image.open(dosya_yolu) as ham:
            # Model tarafındaki ile aynı EXIF düzeltmesi: yan yatmış fotoğrafta yazı okunamaz.
            ham = _kucult(ImageOps.exif_transpose(ham).convert("RGB"))
            en_iyi, en_iyi_skor = [], -1.0
            for mod in ("ham", "gri"):
                satirlar = _satirlari_cikar(_varyant(ham, mod), pytesseract, yapilandirma)
                puan = _skor(satirlar)
                if puan > en_iyi_skor:
                    en_iyi, en_iyi_skor = satirlar, puan
    except Exception:
        # Bozuk görsel, eksik dil dosyası, tesseract hatası: sessizce vazgeç.
        return []

    temiz = _alt_dizeleri_ele(en_iyi)
    temiz.sort(key=lambda s: s[2])  # görseldeki dikey sıra
    return [metin for metin, _, _ in temiz]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python ocr.py <görsel yolu>")
        raise SystemExit(1)
    for satir in yazilari_oku(sys.argv[1]):
        print(satir)
