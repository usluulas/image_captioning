import os
import io
import json
import re    # Regex kütüphanesi (JSON Zırhı)
from collections import deque
from datetime import datetime, timezone
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

# Ollama, 4GB'lık kartta kendine ~2.8 GiB tavan koy uyor (log: available="2.8 GiB" free="3.2 GiB")
# ve model buna sığmadığı için katmanların çoğunu CPU'ya taşıyor (ölçüm: 67%/33% CPU/GPU).
# Bu sadece yavaşlatmıyor, ÇIKTIYI DA BOZUYOR: harry_potter.jpg'de afişte hiç olmayan
# "Voldemort", "Draco" ve "ejderha", sm.jpg'de uydurma "Meryl Streep" çıkıyordu. Tüm katmanlar
# GPU'ya zorlandığında bu uydurmalar kayboldu, afişteki 6 karakter doğru sayıldı.
# GPU'suz makinede (Docker/CPU) parametre sessizce yok sayılıyor — çıktı birebir aynı kalıyor.
# Sığmayan makinelerde ortam değişkeniyle düşürülebilir.
NUM_GPU = int(os.environ.get("NUM_GPU", "34"))  # gemma3:4b'nin katman sayısı

# temperature 0.0 (greedy) modeli deterministik yapar, halüsinasyonu azaltır.
# repeat_penalty tekrar döngülerine karşı güvenlik payı; ama 1.15 fazla agresifti. Greedy
# decoding'de cümlede geçen bir kelime etikette tekrarlanınca ceza doğru token'ı bastırıp
# kelimeyi ortasından bozuyordu: "turuncu" -> "turuncugüç" / "turuncuya" (10 görselin 2'sinde,
# ikisi de cümlede "turuncu" geçtikten sonra). 1.05'te bu kayboldu; yan etki olarak bej arka
# plan rengi kuşa sızmayı bıraktı ve model görselin türünü ("stilize edilmiş") kendiliğinden
# söylemeye başladı. 1.0 fazla düşük: yeni bozulma ("kuşbacakları") ve "stilize yüz" -> "maske"
# gerilemesi üretti. Tekrar döngüsü 1.05'te regresyon setinin hiçbir görselinde görülmedi.
ORTAK_SECENEKLER = {
    'num_ctx': NUM_CTX,
    'num_thread': NUM_THREAD,
    'num_gpu': NUM_GPU,
    'temperature': 0.0,
    'repeat_penalty': 1.05,
}

DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")


# ==========================================
# 1. PROMPT ŞABLONU
# ==========================================
# NOT: Bu prompt oturum boyunca kural kural büyümüştü (847 kelimeye çıkmıştı) ve bir noktada
# 4GB VRAM'deki hesaplama/bellek baskısı üretimi yarıda kestiriyordu (anitkabir.jpg'de done=False).
# Aynı kuralları koruyup daha yoğun/derli toplu (~430 kelime) yeniden yazınca hem çökme ortadan
# kalktı hem de birkaç vaka (isim tanıma, yazı alıntılama, renk) daha iyi çalıştı. Regresyon testi:
# kelebek, gitar, renk/eylem/bağlamsal sızma, yazı alıntılama (BIG BROTHER), oran/hava durumu,
# ikonik isim vakalarının hepsi korundu.
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
# 2. JSON AYIKLAMA VE ŞEMA DOĞRULAMA
# ==========================================
# Prompt en fazla 6 etiket istiyor; kod da aynı sayıyı zorluyor ki ikisi ayrışmasın.
AZAMI_ETIKET = 6
# Prompt 1-3 kelimelik etiket istiyor. Cümle uzunluğunda bir "etiket" arşiv indeksinde
# kullanılamaz; kırpmak yerine atılır (ham yanıt günlükte durduğu için kanıt kaybolmaz).
AZAMI_ETIKET_KELIME = 5
AZAMI_ETIKET_KARAKTER = 40


def turkce_kucult(metin):
    """Python'un .lower()'ı Türkçede yanlış sonuç verir: I -> i yapar, oysa ı olmalı; ayrıca
    "KEDİ".lower() birleşik noktalı "kedi̇" üretir ve "kedi" ile eşleşmez. Bu yüzden hem etiket
    mükerrer elemesinde hem arama terimlerinde bu fonksiyon kullanılır."""
    return metin.replace("İ", "i").replace("I", "ı").lower()


def guvenli_json_ayikla(ham_metin):
    try:
        veri = json.loads(ham_metin.strip())
    except json.JSONDecodeError:
        # format='json' zaten geçerli JSON garanti eder; bu regex sadece ekstra güvenlik katmanı.
        # Greedy (.*) bilinçli: non-greedy, iç içe nesnede ilk '}' ile kesip geçersiz JSON verir.
        match = re.search(r'\{.*\}', ham_metin, re.DOTALL)
        if not match:
            raise ValueError("Yapay zeka yanıtı içinde geçerli bir JSON bloğu bulunamadı.")
        veri = json.loads(match.group(0))

    # Geçerli JSON, doğru şema demek değil: yanıt liste/string/sayı da olabilir. Burada
    # yakalanmazsa aşağıda anlamsız bir AttributeError'a dönüşürdü.
    if not isinstance(veri, dict):
        raise ValueError(f"Yanıt bir JSON nesnesi değil ({type(veri).__name__}).")
    return veri


def metin_dogrula(deger):
    """Şema string bekleyen alanlar için: string değilse ya da boşsa None döner.

    Doğrulama olmadan {"aciklama": 12} çıktıya "12", null ise "None" olarak basılıyordu.
    Bozuk alan sessizce düşürülür; ham yanıt günlüğe yazıldığı için kanıt kaybolmaz.
    """
    if not isinstance(deger, str):
        return None
    temiz = deger.strip()
    return temiz or None


def etiketleri_normallestir(etiketler):
    if not isinstance(etiketler, list):
        return []
    gorulen = set()
    sonuc = []
    for etiket in etiketler:
        temiz = metin_dogrula(etiket)
        if temiz is None:
            continue
        if len(temiz) > AZAMI_ETIKET_KARAKTER or len(temiz.split()) > AZAMI_ETIKET_KELIME:
            continue
        anahtar = turkce_kucult(temiz)
        if anahtar not in gorulen:
            gorulen.add(anahtar)
            sonuc.append(temiz)
        if len(sonuc) >= AZAMI_ETIKET:
            break
    return sonuc


# ==========================================
# 3. ARAMA TERİMLERİ (ÜST KATEGORİ GENİŞLETMESİ)
# ==========================================
# Çıktı bir arşiv sitesinde aranacak. Model "kedi" etiketi üretiyor ama kullanıcı "hayvan" diye
# arayabiliyor; etiketler somut, aramalar geneldir. Bu boşluk MODELE SORULMADAN, sabit sözlükle
# kapatılıyor: deterministik, anlık (model çağrısı yok) ve halüsinasyon riski sıfır.
#
# Bu oturumda üretilmiş 91 gerçek etikete bakılarak tasarlandı:
#   - Türkçede tamlamanın başı SONDA olduğu için son kelimeye bakılır ("mor çiçek" -> "çiçek").
#   - Baş ismin kendisi de arama terimi olur; yoksa "mor çiçek" etiketli fotoğraf düz "çiçek"
#     aramasında çıkmaz.
#   - Ekler YALNIZCA sonuç sözlükte varsa soyulur ("seyirciler"->"seyirci", "Kulesi"->"kule");
#     böylece uydurma kök üretilmez.
#   - Sözlükte olmayan sessizce atlanır. Renkler ("kırmızı"), niteleyiciler ("sarı merkez") ve
#     özel adlar ("Paris", "Hogwarts") böylece kendiliğinden dışarıda kalır.
# BELİRSİZ kelimeler bilerek dışarıda: "oyuncu" (sporcu mu aktör mü), "kale" (yapı mı kaleci mi),
# "karakter", "portre". Arşivde yanlış genişletme, eksik genişletmeden daha zararlıdır.
UST_KATEGORILER = {
    # hayvanlar
    "kedi": ["hayvan", "evcil hayvan", "memeli"],
    "köpek": ["hayvan", "evcil hayvan", "memeli"],
    "fil": ["hayvan", "memeli", "yaban hayatı"],
    "gergedan": ["hayvan", "memeli", "yaban hayatı"],
    "zürafa": ["hayvan", "memeli", "yaban hayatı"],
    "kuş": ["hayvan"],
    "baykuş": ["kuş", "hayvan"],
    "kartal": ["kuş", "hayvan"],
    "böcek": ["hayvan"],
    "kelebek": ["böcek", "hayvan"],
    "arı": ["böcek", "hayvan"],
    "karınca": ["böcek", "hayvan"],
    # bitkiler
    "çiçek": ["bitki", "doğa"],
    "gül": ["çiçek", "bitki"],
    "lavanta": ["çiçek", "bitki"],
    "papatya": ["çiçek", "bitki"],
    "ağaç": ["bitki", "doğa"],
    "sakura": ["ağaç", "bitki"],
    "ot": ["bitki"],
    "çimen": ["bitki", "doğa"],
    "yaprak": ["bitki"],
    "bitki": ["doğa"],
    # insan / spor
    "futbolcu": ["sporcu", "insan"],
    "voleybolcu": ["sporcu", "insan"],
    "sporcu": ["insan"],
    "seyirci": ["insan", "kalabalık"],
    "kadın": ["insan"],
    "adam": ["insan"],
    "çocuk": ["insan"],
    "futbol": ["spor"],
    "voleybol": ["spor"],
    "forma": ["giysi", "spor kıyafeti"],
    "şapka": ["giysi"],
    # nesneler
    "gitar": ["müzik aleti", "enstrüman"],
    "tren": ["taşıt", "ulaşım"],
    "otomobil": ["taşıt", "ulaşım"],
    "araba": ["taşıt", "ulaşım"],
    "patates": ["sebze", "yiyecek"],
    "logo": ["grafik"],
    "yazı": ["metin"],
    # yapı / mekan
    "kule": ["yapı", "anıt"],
    "bina": ["yapı"],
    "duvar": ["yapı"],
    "stadyum": ["yapı", "spor tesisi"],
    "saray": ["yapı", "tarihi eser"],
    "kilise": ["yapı", "ibadethane"],
    "cami": ["yapı", "ibadethane"],
    "müze": ["yapı", "kültür"],
    # sanat / görsel tür
    "fresk": ["resim", "sanat eseri"],
    "tablo": ["resim", "sanat eseri"],
    "afiş": ["grafik", "basılı materyal"],
    "illüstrasyon": ["çizim", "grafik"],
    "çizim": ["grafik"],
    "anime": ["animasyon", "çizgi"],
    # doğa / ortam
    "gökyüzü": ["doğa"],
    "bulut": ["gökyüzü", "doğa"],
    "güneş": ["gökyüzü", "doğa"],
    "deniz": ["su", "doğa"],
    "göl": ["su", "doğa"],
    "nehir": ["su", "doğa"],
    "dalga": ["su", "doğa"],
    "dağ": ["doğa", "coğrafya"],
    "orman": ["doğa", "coğrafya"],
    "bahçe": ["doğa", "peyzaj"],
    "manzara": ["doğa", "peyzaj"],
    "taş": ["doğa", "mineral"],
    "kaya": ["taş", "doğa"],
}

# Soyulacak ekler. Çoğul ve iyelik yetmiyor: betimlemede kelime çekimli geçiyor
# ("futbol formasıyla", "gökyüzünde"), o yüzden hâl ekleri de var ve ekler YİNELEMELİ
# soyuluyor ("formasıyla" -> "forması" -> "forma").
ARAMA_EKLERI = (
    "ları", "leri", "ndan", "nden", "sıyla", "siyle", "yla", "yle", "la", "le",
    "nda", "nde", "dan", "den", "tan", "ten", "da", "de", "ta", "te",
    "nın", "nin", "nun", "nün", "ın", "in", "un", "ün",
    "lar", "ler", "sı", "si", "su", "sü", "ya", "ye", "ı", "i", "u", "ü", "a", "e",
)
ARAMA_AZAMI_DERINLIK = 4


def _kok_adaylari(kelime):
    """Kelimenin kendisinden başlayarak ek soyulmuş adayları GENİŞLİK ÖNCELİKLİ üretir.

    Genişlik öncelikli olması şart: sığ (az soyulmuş) bir aday, derin ve yanlış bir adaydan
    önce denensin. Çağıran ilk sözlük eşleşmesinde durur.

    Güvenlik, adayın sözlükte BİREBİR anahtar olma şartından gelir; sözlükte olmayan bir köke
    inmek hiçbir şey üretmez. Denetim (64 betimleme, 339 eşsiz kelime): 41 ek-soyma eşleşmesi,
    41'i de doğru, 0 yanlış. Kritik olanlar temiz: "gülen"/"gülümseyerek" -> "gül"e BAĞLANMIYOR,
    "sarı" -> "arı"ya bağlanmıyor. Hata yönü güvenli: "yaprağı" (ünsüz yumuşaması) ve "otlar"
    (3 harf alt sınırı) eşleşmiyor, yani yanlış terim eklemek yerine terim eklememe oluyor.

    BAKIM KURALI: UST_KATEGORILER'e KISA anahtar (2-3 harf) eklemekten kaçının, çarpışma riski
    oradan gelir; sözlüğü büyüttükten sonra bu denetimi tekrar koşun.
    """
    gorulen = {kelime}
    kuyruk = deque([(kelime, 0)])
    while kuyruk:
        mevcut, derinlik = kuyruk.popleft()
        yield mevcut
        if derinlik >= ARAMA_AZAMI_DERINLIK:
            continue
        for ek in ARAMA_EKLERI:
            if mevcut.endswith(ek) and len(mevcut) - len(ek) >= 3:
                aday = mevcut[: -len(ek)]
                if aday not in gorulen:
                    gorulen.add(aday)
                    kuyruk.append((aday, derinlik + 1))


def arama_terimleri(etiketler, aciklama=None):
    """Etiketlerden ve betimlemeden üst kategori/genel terimler üretir.

    Etiketlerin kendisi sonuçta YER ALMAZ (mükerrer eleme "mevcut" kümesiyle yapılır).

    Betimleme de taranır çünkü 6 etiket sınırı modeli seçim yapmaya zorluyor: 63 gerçek
    çıktının 12'sinde betimlemede geçen somut bir nesne ("gökyüzü", "forma", "manzara")
    etiketlere girememişti. Betimlemeyi taramak ARAMA'yı ortalama 3.8'den 4.2 terime
    çıkarıyor (en uzun çıktı değişmiyor), yani şişirmeden kaybı kapatıyor.
    """
    mevcut = {turkce_kucult(e) for e in etiketler}
    sonuc = []
    gorulen = set()

    def ekle(terim):
        anahtar = turkce_kucult(terim)
        if anahtar not in mevcut and anahtar not in gorulen:
            gorulen.add(anahtar)
            sonuc.append(terim)

    # Etiketlerde yalnızca baş isme bakılır (Türkçede tamlamanın başı sonda: "mor çiçek"
    # -> "çiçek"); betimlemede ise her kelime aday, çünkü cümlede konum sabit değil.
    adaylar = [turkce_kucult(e).split()[-1] for e in etiketler if e.strip()]
    if aciklama:
        adaylar += re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü'-]+", turkce_kucult(aciklama))

    for kelime in adaylar:
        for aday in _kok_adaylari(kelime):
            if aday in UST_KATEGORILER:
                ekle(aday)
                for terim in UST_KATEGORILER[aday]:
                    ekle(terim)
                break  # ilk eşleşen kök yeter
    return sonuc


# ==========================================
# 4. KOŞU GÜNLÜĞÜ (JSONL)
# ==========================================
# Modeli iyileştirirken en çok ihtiyaç duyulan şey geçmiş koşuların ham verisi: hangi görselde
# hangi gözlemler üretildi, süre nereye gitti, retry oldu mu, kaç token harcandı. Bunların hepsi
# Ollama yanıtında HAZIR geliyor; ayrıca ölçmeye gerek yok.
# İlk ölçümün gösterdiği (neko_gitar, 52.5 sn): yükleme 7.4 + prompt işleme 39.2 + üretim 5.3.
# Yani sürenin ~%75'i görsel işleme, üretim yalnızca ~5 sn. Girdi token'ı farklı görsellerde bile
# sabit (1320) — görsel sabit sayıda token'a çevriliyor, dosya boyutunun süreye etkisi yok.
# Buradan çıkan kural: num_predict/cümle uzunluğu gibi ÇIKTI tarafındaki ayarlar hızı değiştirmez.
#
# Her koşu tek satır JSON (JSONL): ekleme atomik, dosya bozulmadan büyür, satır satır analiz
# edilir. Günlük yazımı BAŞARISIZ OLURSA betimleme akışı etkilenmez — arşiv işi günlükten önemli.
GUNLUK_DOSYASI = os.environ.get("GUNLUK_DOSYASI", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "loglar", "kosular.jsonl"))
GUNLUK_KAPALI = os.environ.get("GUNLUK_KAPALI", "") == "1"


def _yanit_olcumleri(yanit):
    """Ollama yanıtında hazır gelen ölçümler; süreler nanosaniyeden saniyeye çevrilir."""
    def sn(deger):
        return round(deger / 1e9, 2) if isinstance(deger, int) else None

    return {
        "done": yanit.get("done"),
        "done_reason": yanit.get("done_reason"),
        "toplam_sn": sn(yanit.get("total_duration")),
        "yukleme_sn": sn(yanit.get("load_duration")),
        "prompt_isleme_sn": sn(yanit.get("prompt_eval_duration")),
        "uretim_sn": sn(yanit.get("eval_duration")),
        "girdi_token": yanit.get("prompt_eval_count"),
        "cikti_token": yanit.get("eval_count"),
    }


def _gunluge_yaz(kayit):
    if GUNLUK_KAPALI:
        return
    try:
        os.makedirs(os.path.dirname(GUNLUK_DOSYASI), exist_ok=True)
        with open(GUNLUK_DOSYASI, "a", encoding="utf-8") as dosya:
            dosya.write(json.dumps(kayit, ensure_ascii=False) + "\n")
    except Exception:
        # Salt okunur dosya sistemi, izin sorunu, dolu disk: günlük tutulamaz ama program sürer.
        pass


# ==========================================
# 5. GÖRSEL ANALİZ
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
    """Görseli analiz eder ve SÖZLÜK döndürür; biçimlendirme çağıranın işidir.

    Anahtarlar her iki durumda da aynıdır, böylece tüketici KeyError almaz:
        aciklama   (str)  başarısızlıkta "Betimleme çıkarılamadı."
        etiketler  (list) başarısızlıkta boş
        arama      (list) başarısızlıkta boş
        ham_yanit  (str)  modelin ham çıktısı
        hata       (str)  başarıda None; şema/ayrıştırma hatasında "TipAdı: mesaj"

    Durumu ayırt etmek için "hata" alanına bakılır. Bağlantı/Ollama hataları burada
    yakalanmaz, çağırana yükselir.
    """
    gorsel_verisi = gorsel_veri_hazirla(dosya_yolu)
    # Günlük kaydı koşu boyunca doldurulur; retry sayısı "denemeler" listesinin uzunluğudur.
    kayit = {
        "zaman": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dosya": os.path.basename(dosya_yolu),
        "model": MODEL_ADI,
        "secenekler": {**ORTAK_SECENEKLER, "num_predict": NUM_PREDICT},
        "denemeler": [],
    }

    response = _model_cagir(gorsel_verisi)
    kayit["denemeler"].append(_yanit_olcumleri(response))

    # 4GB VRAM'de GPU zaman zaman üretimi ortada kesebiliyor (Ollama watchdog timeout);
    # bu durumda yanıt 'done=False' ve JSON yarım gelir. Bir kez otomatik yeniden dene.
    if not response.get('done'):
        response = _model_cagir(gorsel_verisi)
        kayit["denemeler"].append(_yanit_olcumleri(response))

    # Ayrıştırma başarısız olursa elimizdeki tek kanıt ham yanıt olur; her durumda kaydedilir.
    kayit["ham_yanit"] = response['message']['content']

    try:
        data = guvenli_json_ayikla(response['message']['content'])
        aciklama = metin_dogrula(data.get("aciklama")) or "Betimleme çıkarılamadı."
        etiketler = etiketleri_normallestir(data.get("etiketler", []))

        # "gozlemler" kullanıcıya gösterilmiyor ama hataların çoğu orada doğduğu için
        # günlüğe yazılıyor; sonraki iyileştirmeler bu alana bakarak yapılabilsin.
        kayit["gozlemler"] = data.get("gozlemler")
        kayit["aciklama"] = aciklama
        kayit["etiketler"] = etiketler

        # Sözlük zamanla büyüyeceği için üretilen terimler de kaydediliyor; aksi hâlde günlük,
        # o koşuda gerçekte ne basıldığının kaydı olmaktan çıkardı.
        arama = arama_terimleri(etiketler, aciklama)
        kayit["arama"] = arama
        _gunluge_yaz(kayit)

        return {
            "aciklama": aciklama,
            "etiketler": etiketler,
            "arama": arama,
            "ham_yanit": kayit["ham_yanit"],
            "hata": None,
        }
    except Exception as e:
        kayit["hata"] = f"{type(e).__name__}: {str(e)[:200]}"
        _gunluge_yaz(kayit)
        return {
            "aciklama": "Betimleme çıkarılamadı.",
            "etiketler": [],
            "arama": [],
            "ham_yanit": kayit["ham_yanit"],
            "hata": kayit["hata"],
        }


# ==========================================
# 6. ANA TERMİNAL İŞLEYİŞİ
# ==========================================
# gorseli_betimle() veri döndürür, biçimlendirme buraya ait: aynı sonucu bir arayüz ya da
# toplu işleme script'i kendi istediği gibi sunabilsin diye ikisi ayrıldı.
def sonucu_bicimlendir(sonuc):
    """Betimleme sözlüğünü terminale basılacak metne çevirir."""
    if sonuc["hata"]:
        return (f"[HATA] Yapay zeka istenen formatta yanıt veremedi "
                f"(Sebep: {sonuc['hata'][:50]}).\n\nHam Yanıt:\n{sonuc['ham_yanit']}")

    bolumler = [
        f"BETİMLEME:\n{sonuc['aciklama']}",
        f"ETİKETLER:\n{', '.join(sonuc['etiketler'])}",
    ]
    if sonuc["arama"]:  # Sözlükte karşılık yoksa başlık hiç basılmaz.
        bolumler.append(f"ARAMA:\n{', '.join(sonuc['arama'])}")
    return "\n\n".join(bolumler)


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

        sonuc = gorseli_betimle(dosya_yolu)

        print("\n" + "=" * 60)
        print(f"--- DOSYA: {os.path.basename(dosya_yolu)} ---")
        print("=" * 60)
        print(sonucu_bicimlendir(sonuc))
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[HATA] Bir hata oluştu: {str(e)}")
        print(f"Lütfen arka planda Ollama'nın çalıştığından ('ollama run {MODEL_ADI}') emin olun.")


if __name__ == "__main__":
    main()
