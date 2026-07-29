# image_captioning

Bir fotoğrafı yerel bir görsel-dil modeliyle (Ollama üzerinde `gemma3:4b-it-qat`) analiz edip
Türkçe tek cümlelik betimleme ve etiketler üreten terminal uygulaması. Görselde okunabilir bir
yazı varsa onu da ayrı bir başlık altında verir.

---
## Docker olmadan çalıştırma

1. [Ollama](https://ollama.com/download)'yı kurun.
2. Modeli indirin: `ollama pull gemma3:4b-it-qat`
3. Bağımlılıkları kurun: `pip install -r requirements.txt`
4. (İsteğe bağlı) Yazı okuma için [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)'ı
   kurun — aşağıya bakın.
5. Çalıştırın: `python image_captioning.py`

### Gereken kütüphaneler

3. adımdaki komut aşağıdakilerin tamamını kurar; tek tek kurmak isterseniz sürümler
`requirements.txt` içinde sabitlenmiştir.

| Kütüphane | Sürüm | Ne işe yarar |
|---|---|---|
| `ollama` | 0.6.2 | Ollama sunucusuna görseli ve promptu gönderen Python istemcisi |
| `Pillow` | 12.3.0 | Görseli açar, EXIF'e göre döndürmeyi düzeltir, PNG baytlarına çevirir |
| `pillow-heif` | 1.4.0 | iPhone'un `.heic`/`.heif` formatını Pillow'a tanıtır |
| `pytesseract` | 0.3.13 | Tesseract'ı Python'dan çağırır (yazı okuma) |

Tek komutla:

```bash
pip install ollama==0.6.2 Pillow==12.3.0 pillow-heif==1.4.0 pytesseract==0.3.13
```

Bunlar yalnızca Python bağımlılıklarıdır; **Ollama'nın kendisi ayrı bir programdır** (1. adım) ve
`pip` ile kurulmaz. Docker ile çalıştırıyorsanız bu adımların hiçbirini yapmanıza gerek yoktur —
imaj bunları kendisi kurar.

### Yazı okuma (Tesseract) — isteğe bağlı

Görseldeki yazılar `YAZILAR` başlığı altında ayrıca listelenir. Bu iş modele değil,
[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) adlı ayrı bir OCR motoruna yaptırılır
(sebebi aşağıda). `pytesseract` yalnızca ince bir köprüdür; **Tesseract'ın kendisi de ayrı bir
sistem programıdır.**

Windows'ta:

```bash
winget install UB-Mannheim.TesseractOCR
```

Kurulum yalnızca İngilizce dil verisiyle gelir. Türkçe metinler (ç, ğ, ı, İ, ö, ş, ü) için
`tur.traineddata` dosyasını [tessdata_best](https://github.com/tesseract-ocr/tessdata_best)
deposundan indirip bir `tessdata` klasörüne koyun ve o klasörü `TESSDATA_DIZINI` ortam
değişkeniyle belirtin (varsayılan: `%LOCALAPPDATA%\tessdata`). `eng.traineddata` ve
`osd.traineddata` dosyalarını da Tesseract'ın kurulum klasöründen aynı yere kopyalayın.

**Tesseract kurulu değilse program normal çalışır**, sadece `YAZILAR` bölümü hiç çıkmaz.

İlgili ortam değişkenleri: `TESSERACT_YOLU`, `TESSDATA_DIZINI`, `OCR_DILLERI` (varsayılan
`tur+eng`), `OCR_GUVEN_ESIGI` (varsayılan `60`).

#### Neden ayrı bir OCR motoru?

Modelden yazıları da listelemesi istendiğinde iki sorun çıktı: haber görselindeki 9 satırlık fiyat
tablosunu hiç alamadı, ve **yazısız görsellere metin uydurdu** (bir dağ manzarasına
"Lago di Braies", bir voleybol fotoğrafına "Orhan Kemal" yazdı). Tesseract okuyamadığında boş
döner — arşiv çıktısı için belirleyici olan bu.

İkisinin güçlü olduğu yerler farklı:

| | Model | Tesseract |
|---|---|---|
| Yoğun düz metin (tablo, belge) | Alamıyor | 9 fiyatın 8'ini doğru okudu |
| Stilize afiş başlığı (`LOST`, `SPIDER-MAN`) | Okuyor | Okuyamıyor |
| Yazı olmayan görsel | Uydurabiliyor | Boş dönüyor |

Bu yüzden `YAZILAR` bölümü yalnızca Tesseract'tan gelir; film/dizi adı gibi stilize başlıklar
zaten `BETİMLEME` cümlesinin içinde geçer.

---

## Dosyalar

| Dosya | Görevi |
|---|---|
| `image_captioning.py` | Ana program: prompt, görsel hazırlama, model çağrısı, terminal arayüzü |
| `ocr.py` | Görseldeki yazıyı Tesseract ile okur (`YAZILAR` bölümü). Tesseract yoksa boş döner |
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

