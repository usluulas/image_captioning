# image_captioning

Bir fotoğrafı yerel bir görsel-dil modeliyle (Ollama üzerinde `gemma3:4b-it-qat`) analiz edip
Türkçe tek cümlelik betimleme, etiketler ve arama terimleri üreten terminal uygulaması.

---
## Docker olmadan çalıştırma

1. [Ollama](https://ollama.com/download)'yı kurun.
2. Modeli indirin: `ollama pull gemma3:4b-it-qat`
3. Bağımlılıkları kurun: `pip install -r requirements.txt`
4. Çalıştırın: `python image_captioning.py`

### Gereken kütüphaneler

3. adımdaki komut aşağıdakilerin tamamını kurar; tek tek kurmak isterseniz sürümler
`requirements.txt` içinde sabitlenmiştir.

| Kütüphane | Sürüm | Ne işe yarar |
|---|---|---|
| `ollama` | 0.6.2 | Ollama sunucusuna görseli ve promptu gönderen Python istemcisi |
| `Pillow` | 12.3.0 | Görseli açar, EXIF'e göre döndürmeyi düzeltir, PNG baytlarına çevirir |
| `pillow-heif` | 1.4.0 | iPhone'un `.heic`/`.heif` formatını Pillow'a tanıtır |

Tek komutla:

```bash
pip install ollama==0.6.2 Pillow==12.3.0 pillow-heif==1.4.0
```

Bunlar yalnızca Python bağımlılıklarıdır; **Ollama'nın kendisi ayrı bir programdır** (1. adım) ve
`pip` ile kurulmaz. Docker ile çalıştırıyorsanız bu adımların hiçbirini yapmanıza gerek yoktur —
imaj bunları kendisi kurar.

### Arama terimleri

Çıktıdaki `ARAMA` bölümü, etiketlerden türetilen daha genel terimlerdir. Model somut etiket
üretir ("kedi", "mor çiçek"), oysa arşivde arama yapan kişi genel arar ("hayvan", "çiçek").
Örnek:

```
ETİKETLER: mor çiçek, siyah böcek, sarı merkez, kuru ot, gri taş, pembe kenar
ARAMA:     çiçek, bitki, doğa, böcek, hayvan, ot, taş, mineral
```

Bu iş **modele sorulmaz**; `UST_KATEGORILER` sözlüğünden deterministik olarak üretilir. Model
çağrısı olmadığı için ek süre getirmez ve uydurma terim üretemez. Türkçede tamlamanın başı sonda
olduğu için son kelimeye bakılır ("mor çiçek" → "çiçek"); çoğul ve iyelik ekleri yalnızca sonuç
sözlükte varsa soyulur ("seyirciler" → "seyirci", "Eiffel Kulesi" → "kule").

Sözlükte karşılığı olmayan etiketler sessizce atlanır — renkler, niteleyiciler ("sarı merkez")
ve özel adlar ("Paris", "Hogwarts") böylece dışarıda kalır. Belirsiz kelimeler ("oyuncu",
"kale", "karakter") bilerek eklenmemiştir: arşivde yanlış genişletme, eksik genişletmeden daha
zararlıdır. Sözlük `image_captioning.py` içindedir ve zamanla büyütülmek üzere tasarlanmıştır.

### Koşu günlüğü

Her koşu `loglar/kosular.jsonl` dosyasına tek satır JSON olarak eklenir. Amaç modeli
iyileştirirken elde veri olması: kullanıcıya gösterilmeyen `gozlemler` alanı, ham model yanıtı,
süre dökümü, token sayıları ve retry sayısı (`denemeler` listesinin uzunluğu).

Süre dökümü Ollama'dan hazır gelir ve öğreticidir — örnek bir koşu:

```
toplam 52.5 sn = yükleme 7.4 + prompt işleme 39.2 + üretim 5.3
girdi token: 1320   çıktı token: 160
```

Sürenin ~%75'i görselin işlenmesine gidiyor, üretim yalnızca ~5 saniye. Girdi token sayısı
farklı görsellerde bile sabit; görsel sabit sayıda token'a çevriliyor, dosya boyutunun süreye
etkisi yok. Pratik sonucu: cümle uzunluğu veya `num_predict` gibi **çıktı tarafındaki ayarlar
hızı değiştirmez**.

Günlük yazımı başarısız olursa (salt okunur dosya sistemi, izin sorunu) program normal çalışmaya
devam eder, sessizce günlük tutmaz. Ortam değişkenleri: `GUNLUK_DOSYASI` ile yol değiştirilir,
`GUNLUK_KAPALI=1` ile tamamen kapatılır. `loglar/` klasörü `.gitignore` kapsamındadır.

Docker'da günlük container içine yazılır ve container silinince kaybolur; kalıcı olması için
`docker-compose.yml`'deki `betimleyici` servisine bir volume eklemek gerekir.

### Görseldeki yazılar

Ayrı bir yazı okuma (OCR) adımı yoktur. Afiş başlığı, tabela, logo gibi okunabilir yazıları model
kendisi görüp `BETİMLEME` cümlesinin içinde orijinal dilinde alıntılar (ör. "'Big Brother is
watching you' yazısı"). Yoğun düz metni (tablo, belge) güvenilir biçimde çıkaramaz; bu tür
görsellerde metnin tamamını beklemeyin.

---

## Dosyalar

| Dosya | Görevi |
|---|---|
| `image_captioning.py` | Ana program: prompt, görsel hazırlama, model çağrısı, terminal arayüzü |
| `loglar/kosular.jsonl` | Koşu günlüğü (otomatik oluşur, repoya girmez) |
| `requirements.txt` | Python bağımlılıkları (sürümler sabitlenmiştir) |
| `Dockerfile` | Python ortamının tarifi |
| `entrypoint.sh` | Container açılışında Ollama'yı bekler, model yoksa indirir |
| `docker-compose.yml` | Ollama sunucusu + uygulama container'ını birlikte ayağa kaldırır |
| `docker-compose.gpu.yml` | NVIDIA GPU kullanımı için ek ayar dosyası |
| `docker-compose.mac.yml` | macOS'ta Mac'e kurulu Ollama'ya bağlanan alternatif yapılandırma |

## Docker ile çalıştırma (önerilen — hiçbir şey kurmanız gerekmez)

### 1. Docker Desktop'ı kurun

- <https://www.docker.com/products/docker-desktop/> adresinden **Docker Desktop for Windows**'u indirin.
- Kurulumda **"Use WSL 2 instead of Hyper-V"** seçeneği işaretli kalsın.
- Kurulum bittikten sonra **bilgisayarı yeniden başlatın**.
- Docker Desktop'ı açın ve sol alttaki durum göstergesi yeşil **"Engine running"** olana kadar bekleyin.

Kontrol için terminalde:

```bash
docker --version
```

### 2. Projeyi çalıştırın

Proje klasöründe terminal açıp:

```bash
docker compose run --rm betimleyici
```

İlk çalıştırma uzun sürer: Docker imajı hazırlanır ve model (~4 GB) indirilir. Sonraki
çalıştırmalar saniyeler içinde başlar — model kalıcı bir Docker volume'ünde saklanır.

### 3. Fotoğrafı verin

Betimlenecek fotoğrafları proje içindeki **`gorseller/`** klasörüne kopyalayın.
Program yol sorduğunda `/gorseller/` ön ekiyle yazın:

```
Lütfen betimlenecek fotoğrafın yolunu girin:
> /gorseller/kelebek.jpg
```

Desteklenen formatlar: `.jpg .jpeg .png .webp .bmp .heic .heif`

### 4. (İsteğe bağlı) NVIDIA ekran kartı varsa

GPU kullanmak analizi 10-20 kat hızlandırır. NVIDIA kartı olan makinelerde:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm betimleyici
```

Ekran kartı yoksa bu komutu **kullanmayın**; normal komut CPU üzerinde çalışır
(fotoğraf başına yaklaşık 1-3 dakika, sonuç aynıdır).

### 5. (İsteğe bağlı) Mac kullanıyorsanız

Docker, macOS'ta Apple Silicon'ın GPU'suna (Metal) erişemez — normal komut Mac'te de yalnızca
CPU kullanır ve yavaş çalışır. Tam hız için Ollama'yı Mac'e normal uygulama olarak kurup
container'ı ona bağlayabilirsiniz:

1. <https://ollama.com/download> adresinden Ollama'yı kurun ve çalıştırın.
2. Ollama'nın container'dan gelen bağlantıları kabul etmesi için terminalde bir kez:

   ```bash
   launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
   ```

   Ardından Ollama uygulamasını kapatıp yeniden açın.
3. Programı şu komutla çalıştırın:

   ```bash
   docker compose -f docker-compose.mac.yml run --rm betimleyici
   ```

Bu yolda Ollama container'ı hiç başlatılmaz; model Mac'in GPU'sunda çalışır, Python tarafı
yine container içinde izole kalır. Kolaylık isteniyorsa normal komut da Mac'te çalışır,
sadece yavaştır.

### Kapatma

Program bittikten sonra arka planda kalan Ollama sunucusunu durdurmak için:

```bash
docker compose down
```

### Bilinen sınırlama: çıktı birebir aynı olmayabilir

`temperature=0.0` ayarı modeli deterministik yapar, ancak bu determinizm **aynı donanım ve aynı
çalıştırma ortamı** için geçerlidir. Aynı görsel Docker içinde (CPU) ve Docker dışında (GPU)
çalıştırıldığında betimleme anlamca aynı, kelimesi kelimesine farklı çıkabilir — çünkü hesaplama
farklı sayısal yollardan geçer. Örnek:

| Ortam | Çıktı |
|---|---|
| Docker (CPU) | "Beyaz bir Fender gitarın yanında, desenli bir kedi oturan bir odada durmaktadır." |
| Yerel (GPU) | "Beyaz bir Fender Telecaster gitarın yanında bembeyaz ve gri-siyah çizgili bir kedi oturmuş durumda, arka planda büyük bir tablo ve televizyon bulunuyor." |

Docker'ın çözdüğü şey **"hiç çalışmıyor"** problemidir; **"kelimesi kelimesine aynı çıktı"** garantisi
vermez.

---

