# image_captioning

Bir fotoğrafı yerel bir görsel-dil modeliyle (Ollama üzerinde `gemma3:4b-it-qat`) analiz edip
Türkçe tek cümlelik betimleme ve etiketler üreten uygulama. Tarayıcıda açılan bir arayüzü
(`arayuz.py`) ve aynı işi yapan bir terminal sürümü (`image_captioning_ui.py`) vardır.

---
## Docker olmadan çalıştırma

1. [Ollama](https://ollama.com/download)'yı kurun.
2. Modeli indirin: `ollama pull gemma3:4b-it-qat`
3. Bağımlılıkları kurun: `pip install -r requirements.txt`
4. Çalıştırın:
   - Arayüz için: `python arayuz.py` → tarayıcıda <http://127.0.0.1:7860> açılır.
   - Terminal sürümü için: `python image_captioning_ui.py`

### Arayüz internete açılmıyor

Arayüz tarayıcıda açılsa da uygulama **tamamen çevrimdışıdır**; internet bağlantısı olmayan
bir bilgisayarda da çalışır. Ayrıntılar:

- Sunucu yalnızca `127.0.0.1` adresini dinler — aynı ağdaki başka cihazlar bile erişemez.
- Fotoğraflar ve betimlemeler bilgisayardan **hiç çıkmaz**; model de yerelde (Ollama) çalışır.
- Gradio'nun genel paylaşım bağlantısı (`share=True`) kullanılmaz, kullanım istatistiği
  gönderimi (`GRADIO_ANALYTICS_ENABLED=False`) kapatılmıştır.
- Arayüzün tüm dosyaları (yazı tipleri dahil) yerel sunucudan gelir, hiçbir dış siteden
  içerik çekilmez.

Arayüzde fotoğrafı kutuya sürükleyip **Betimle** düğmesine basmanız yeterli. Betimleme ve
etiketler ayrı kutularda çıkar, her birinin sağ üstündeki kopyalama simgesiyle tek tıkla
panoya alınıp arşiv sitesine yapıştırılabilir.

### Gereken kütüphaneler

3. adımdaki komut aşağıdakilerin tamamını kurar; tek tek kurmak isterseniz sürümler
`requirements.txt` içinde sabitlenmiştir.

| Kütüphane | Sürüm | Ne işe yarar |
|---|---|---|
| `ollama` | 0.6.2 | Ollama sunucusuna görseli ve promptu gönderen Python istemcisi |
| `Pillow` | 12.3.0 | Görseli açar, EXIF'e göre döndürmeyi düzeltir, PNG baytlarına çevirir |
| `pillow-heif` | 1.4.0 | iPhone'un `.heic`/`.heif` formatını Pillow'a tanıtır |
| `gradio` | 6.20.0 | Tarayıcıda açılan arayüzü oluşturur (yalnızca `arayuz.py` için gerekir) |

Tek komutla:

```bash
pip install ollama==0.6.2 Pillow==12.3.0 pillow-heif==1.4.0 gradio==6.20.0
```

Bunlar yalnızca Python bağımlılıklarıdır; **Ollama'nın kendisi ayrı bir programdır** (1. adım) ve
`pip` ile kurulmaz. Docker ile çalıştırıyorsanız bu adımların hiçbirini yapmanıza gerek yoktur —
imaj bunları kendisi kurar.

---

## Dosyalar

| Dosya | Görevi |
|---|---|
| `image_captioning_ui.py` | Çekirdek: prompt, görsel hazırlama, model çağrısı + terminal sürümü |
| `arayuz.py` | Tarayıcıda açılan grafik arayüz; model işini çekirdek dosyaya devreder |
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
docker compose up
```

İlk çalıştırma uzun sürer: Docker imajı hazırlanır ve model (~4 GB) indirilir. Sonraki
çalıştırmalar saniyeler içinde başlar — model kalıcı bir Docker volume'ünde saklanır.

### 3. Arayüzü açın

Terminalde `Running on http://0.0.0.0:7860` satırını gördükten sonra tarayıcıda:

<http://localhost:7860>

Fotoğrafı kutuya sürükleyin, **Betimle** düğmesine basın. Docker ile çalıştırırken
fotoğrafın `gorseller/` klasöründe olması **gerekmez** — dosyayı bilgisayarınızın
herhangi bir yerinden yükleyebilirsiniz.

Desteklenen formatlar: `.jpg .jpeg .png .webp .bmp .heic .heif`

### 4. (İsteğe bağlı) NVIDIA ekran kartı varsa

GPU kullanmak analizi 10-20 kat hızlandırır. NVIDIA kartı olan makinelerde:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
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
   docker compose -f docker-compose.mac.yml up
   ```

Bu yolda Ollama container'ı hiç başlatılmaz; model Mac'in GPU'sunda çalışır, Python tarafı
yine container içinde izole kalır. Kolaylık isteniyorsa normal komut da Mac'te çalışır,
sadece yavaştır.

### Kapatma

Arayüzü durdurmak için terminalde `Ctrl+C`. Ardından arka planda kalan Ollama sunucusunu
da durdurmak için:

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

