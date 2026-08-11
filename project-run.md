# راهنمای کامل اجرای پروژه

> این نسخه با کد فعلی (`main.py` / `pipeline.py`) هم‌خوانی داده شده؛ چند مقدار پیش‌فرض و چند آرگومان که در نسخه‌ی قبلی این فایل جا افتاده یا اشتباه بود، اصلاح شده‌اند.

## نحوه اجرا

```bash
python main.py --mode file --file path/to/image.jpg [آرگومان‌های دیگر]
python main.py --mode camera [آرگومان‌های دیگر]
```

در حالت `camera`: کلید `s` یک اسکن انجام می‌دهد، کلید `q` از حلقه خارج می‌شود.

## آرگومان‌های عمومی

| آرگومان | مقدار پیش‌فرض | گزینه‌ها | توضیح |
|---------|---------------|----------|-------|
| `--mode` | `camera` | `camera`, `file` | حالت اجرا: پردازش یک فایل یا استریم زنده دوربین |
| `--file` | - | مسیر فایل | اجباری وقتی `--mode file` است |
| `--no-fp16` | خاموش | flag | غیرفعال کردن fp16 در همه‌ی مدل‌ها (obstacle، text detection، text recognition)؛ دقت کمی بالاتر ولی سرعت پایین‌تر روی GPU. روی CPU اصلاً fp16 فعال نمی‌شود، این فلگ بی‌اثر است. |
| `--realtime-budget` | `2.5` | عدد اعشاری (ثانیه) | سقف زمانی مطلوب برای هر فریم. همیشه در لاگ گزارش می‌شود؛ **اگر `--adaptive-resolution` هم فعال باشد**، این عدد مبنای تصمیم برای کم/زیاد کردن رزولوشن استنتاج obstacle/corner detector هم قرار می‌گیرد. |
| `--adaptive-resolution` | خاموش | flag | اگر فعال باشد، پایپ‌لاین بعد از هر فریم بر اساس `--realtime-budget` رزولوشن استنتاج obstacle detector و corner detector را خودکار کم یا زیاد می‌کند (بین ۳۲۰ تا ۱۶۰۰ پیکسل). پیش‌فرض خاموش است تا زمان‌بندی/نتایج قابل تکرار بمانند. |
| `--debug` | خاموش | flag | ذخیره‌ی تصاویر مراحل میانی روی دیسک (لیست کامل در بخش «خروجی‌های حالت Debug»). |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | سطح لاگ‌ها. `DEBUG` جزئیات بیشتری از جمله لاگ هر آبجکت تشخیص‌داده‌شده و هر کلمه‌ی بازشناسی‌شده را چاپ می‌کند. |
| `--quiet` | خاموش | flag | خاموش کردن کامل لاگ‌ها (فقط خروجی نهایی `print` می‌شود). |

## آرگومان‌های تشخیص مانع/صفحه (Obstacle Detector)

| آرگومان | مقدار پیش‌فرض | گزینه‌ها | توضیح |
|---------|---------------|----------|-------|
| `--obstacle-backbone` | `r18vd` | `r18vd`, `r34vd`, `r50vd`, `r101vd` | مدل: RT-DETRv2 — تشخیص صفحه لپ‌تاپ/مانیتور (کلاس `laptop`/`tvmonitor`، برای برش اولیه) و مانع (کلاس `person`) روی آن. |

> پیش‌فرض واقعی پروژه سبک‌ترین بک‌بون (`r18vd`) است، نه `r34vd`؛ برای دقت بیشتر آن را دستی بدهید.

### مقایسه‌ی بک‌بون‌ها

| بک‌بون | دقت | سرعت | حجم |
|--------|-----|------|-----|
| `r18vd` | متوسط (پیش‌فرض) | خیلی سریع | کم |
| `r34vd` | خوب، تعادل | سریع | متوسط |
| `r50vd` | بالا | متوسط | متوسط-زیاد |
| `r101vd` | بالاترین | کند | زیاد |

> آستانه‌های تشخیص صفحه/مانع (`screen_threshold=0.3`, `obstacle_threshold=0.5`) و نسبت مساحت تداخل برای «انسداد» (`obstruction_area_ratio=0.01`) در سطح کد ثابت هستند و فعلاً آرگومان CLI ندارند؛ اگر لازم شد باید در `strategies/obstacle.py` و فراخوانی سازنده‌اش در `main.py` تغییر کنند.

## آرگومان‌های تشخیص گوشه سند (Corner Detection)

این مرحله در حال حاضر آرگومان CLI ندارد و همیشه از `DocAligner_RealWeightsDetector` استفاده می‌شود (ورودی: کل فریم یا کراپ اولیه‌ی خروجی مرحله‌ی قبل؛ خروجی: ۴ گوشه‌ی سند/مانیتور). اگر مدل گوشه‌ای پیدا نکند، به‌ترتیب به contour-detection و سپس (در صورت وجود bbox صفحه) به bbox خام صفحه سقوط می‌کند. با `--debug` تصویر دقیق ورودی مدل هم ذخیره می‌شود.

## آرگومان‌های Rectification (تصحیح پرسپکتیو)

این مرحله هم CLI ندارد؛ همیشه `DynamicRectifier` (بر پایه‌ی `kornia`) با `max_output_dim=5760` استفاده می‌شود و ابعاد خروجی را متناسب با نسبت طول/عرض سند واقعی محاسبه می‌کند.

## آرگومان‌های تشخیص باکس متن (Text Detection)

| آرگومان | مقدار پیش‌فرض | گزینه‌ها | توضیح |
|---------|---------------|----------|-------|
| `--detector` | `dbnet` | `dbnet`, `ensemble`, `craft`, `east`, `full` | مدل تشخیص متن. توضیح گزینه‌ها در ادامه. |
| `--east-model` | `frozen_east_text_detection.pb` | مسیر فایل | وزن مدل EAST (فقط برای `--detector east` یا `full`) |

> پیش‌فرض واقعی پروژه `dbnet` (سریع‌ترین) است، نه `ensemble`.

### توضیح گزینه‌های `--detector`

- **`dbnet`**: فقط یک مدل docTR با معماری DBNet (`db_resnet50`). سریع‌ترین، دقت خوب روی متن چاپی تمیز. **پیش‌فرض پروژه.**
- **`ensemble`**: سه معماری docTR (`db_resnet50`, `db_mobilenet_v3_large`, `linknet_resnet18`) با هم اجرا و باکس‌ها با IoU ادغام می‌شوند. دقت بالاتر از `dbnet` تنها، مخصوصاً برای متن‌های ریز یا کیفیت پایین.
- **`craft`**: مدل CRAFT، قوی در متن‌های خمیده/نامنظم و کاراکتر-محور.
- **`east`**: مدل EAST، خیلی سریع ولی دقت پایین‌تر روی صحنه‌های شلوغ؛ نیازمند دانلود دستی فایل `.pb` (پایین همین صفحه توضیح داده شده).
- **`full`**: ترکیب `ensemble` + `craft` + `east` با رأی‌گیری IoU. بیشترین دقت و ریکال، ولی کندترین.

پس از تشخیص، باکس‌ها یک بار با NMS (`torchvision.ops.nms`, IoU=0.5) و سپس با الگوریتم داخلی `smart_merge_boxes` بر اساس ستون/ردیف و فاصله‌ی افقی بازچینی/ادغام می‌شوند تا کلمات نزدیک هم در یک واحد قرار بگیرند (بدون این‌که در حالت پیش‌فرض چیزی merge شود؛ merge واقعی فقط اگر خود شما این تابع را با `enable_merging=True` صدا بزنید فعال است — در مسیر فعلی پایپ‌لاین با تنظیمات پیش‌فرض فراخوانی می‌شود).

### مقایسه‌ی مدل‌های تشخیص متن

| مدل | دقت تقریبی | سرعت | مناسب برای |
|-----|------------|------|------------|
| DBNet (docTR) | ~۹۵٪ Hmean (ICDAR2015) | زیاد | تعادل کلی، پیش‌فرض |
| CRAFT | بالا در متن خمیده | متوسط | متن نامنظم/دستنویس |
| EAST | متوسط | خیلی زیاد | real-time محض، صحنه ساده |
| Ensemble (چند docTR) | بالاتر از تک‌مدل | متوسط-زیاد | دقت بهتر با سرعت قابل قبول |
| Full (همه با هم) | بیشترین دقت/ریکال | کند | آفلاین یا وقتی دقت اولویت اصلی است |

## آرگومان‌های تشخیص متن (Text Recognition)

| آرگومان | مقدار پیش‌فرض | گزینه‌ها | توضیح |
|---------|---------------|----------|-------|
| `--recognizer` | `fast` | `trocr`, `fast`, `ppocrv5` | مدل اصلی تشخیص متن |
| `--trocr-size` | `base` | `small`, `base`, `large` | (فقط برای `--recognizer trocr`) |
| `--beams` | `1` | عدد صحیح | (فقط برای `trocr`) تعداد beam search |
| `--fast-arch` | `parseq` | `parseq`, `master`, `crnn_mobilenet_v3_small`, `crnn_vgg16_bn`, `crnn_mobilenet_v3_large`, `vitstr_small` | (فقط برای `--recognizer fast`) |
| `--ppocr-server-url` | `http://127.0.0.1:5005` | آدرس | آدرس میکروسرویس PP-OCRv5 (فقط برای `--recognizer ppocrv5`) |
| `--fallback-recognizer` | `none` | `none`, `trocr-small`, `ppocrv5`, `parseq`, `master`, `crnn_mobilenet_v3_small`, `crnn_vgg16_bn`, `crnn_mobilenet_v3_large`, `vitstr_small` | یک مدل دوم که **فقط** روی کلمات با کیفیت پایین اجرا می‌شود (نه کل تصویر). |

> ⚠️ **پیش‌نیاز `--recognizer ppocrv5`:** چون PP-OCRv5 در یک محیط/پردازه‌ی جدا اجرا می‌شود (به‌خاطر تداخل نسخه‌ای با بقیه‌ی پروژه)، قبل از اجرای `main.py` باید سرویسش روشن باشد (بخش «راه‌اندازی میکروسرویس PP-OCRv5» را ببینید). اگر روشن نباشد، همان لحظه‌ی ساخته‌شدن recognizer یک خطای واضح می‌گیرید که بگویید سرویس را اول اجرا کنید — نه شکست خاموش وسط پردازش.

### `--fallback-recognizer` دقیقاً چه‌کار می‌کند؟

بعد از این‌که `--recognizer` اصلی همه‌ی کلمات یک خط را خواند، پایپ‌لاین کلماتی را که «مشکوک» تشخیص می‌دهد دوباره فقط با مدل fallback می‌خواند:

- کلماتی که مدل اصلی برایشان `conf < 0.97` گزارش کرده (این فقط برای `fast` و `ppocrv5` معنا دارد چون `trocr` امتیاز اطمینان برنمی‌گرداند و طبق پیش‌فرض `1.0` در نظر گرفته می‌شود).
- کلماتی که «از نظر هندسی مشکوک»‌اند: طول متن خروجی نسبت به عرض باکس تشخیص‌داده‌شده غیرمنطقی زیاد است (نشانه‌ی متن قاطی‌شده/غلط).

نتیجه‌ی fallback فقط اگر متن غیرخالی برگرداند جایگزین متن اصلی می‌شود. این ویژگی برای صحنه‌های آزمایشگاهی که دقت بالاتر از سرعت اهمیت دارد مفید است؛ مثلاً `--recognizer fast --fallback-recognizer trocr-small` یا `--recognizer fast --fallback-recognizer ppocrv5`.

### مقایسه‌ی مدل‌های تشخیص متن

| مدل/معماری | دقت تقریبی | سرعت | حجم مدل | مناسب برای |
|-----------|------------|------|---------|------------|
| TrOCR-small | خوب | متوسط | کم | تعادل سبک |
| TrOCR-base | بالا | متوسط-کند | متوسط | دقت خوب عمومی |
| TrOCR-large | بسیار بالا | کند | زیاد | دقت حداکثری، غیر real-time |
| PARSeq (`fast`) | ~۹۴٪ | زیاد | متوسط | تعادل خوب سرعت/دقت |
| MASTER (`fast`) | بالا | متوسط | متوسط-زیاد | دقت بالاتر از CRNN |
| ViTSTR-small (`fast`) | خوب | زیاد | کم | سبک و سریع |
| CRNN-mobilenet-small (`fast`) | متوسط | خیلی زیاد | خیلی کم | ادوات محدود/موبایل |
| CRNN-mobilenet-large (`fast`) | خوب | زیاد | کم | تعادل سبک |
| CRNN-vgg16 (`fast`) | خوب-بالا | متوسط | متوسط | دقت بهتر، کمی کندتر |
| PP-OCRv5 | بسیار بالا | خیلی زیاد | خیلی کم (~۷۰M) | بهترین گزینه‌ی کلی برای real-time با دقت بالا |

## خروجی‌های حالت Debug (`--debug`)

وقتی `--debug` فعال باشد، فایل‌های زیر در دایرکتوری جاری (کنار `main.py`) ذخیره می‌شوند:

| فایل | تولیدشده در | توضیح |
|------|---------------|-------|
| `debug_corner_model_input.jpg` | corner detector | دقیقاً همان تصویری که به DocAligner داده شده (بعد از کراپ/ریسایز اولیه). |
| `step1_corners_detected.jpg` | pipeline | تصویر کامل با ۴ گوشه‌ی تشخیص‌داده‌شده و چندضلعی سند. |
| `step1_flattened.jpg` | pipeline | تصویر بعد از تصحیح پرسپکتیو (rectify). |
| `step2_obstacle_detected.jpg` | pipeline / obstacle detector | فقط اگر مانعی روی سند/صفحه تشخیص داده شود؛ نمایش bbox مانیتور (سبز) و مانع (قرمز). در این حالت فریم رد می‌شود و پردازش OCR ادامه پیدا نمی‌کند. |
| `step3_text_boxes.jpg` | pipeline | همه‌ی باکس‌های متنی تشخیص‌داده‌شده به‌همراه `word_id`. |
| `step4_reconstructed.jpg` | pipeline | بازسازی بصری متن نهایی (هر کلمه داخل باکس خودش، شبیه پیش‌نمایش OCR). |

اگر `step2_obstacle_detected.jpg` تولید شود یعنی پردازش در همان‌جا متوقف شده و بقیه‌ی فایل‌ها (step3/step4) در آن اجرا ساخته نمی‌شوند.

## راه‌اندازی میکروسرویس PP-OCRv5

پکیج‌های `paddlepaddle` و `paddleocr` هرگز نباید داخل محیط اصلی پروژه (`torch_env`) نصب شوند؛ نسخه‌های `torch`/`opencv`/`huggingface-hub` را می‌شکنند. این‌ها فقط در یک محیط conda کاملاً جدا نصب و به‌صورت یک میکروسرویس Flask اجرا می‌شوند:

```bash
conda create -n ppocr_env python=3.10 -y
conda activate ppocr_env
pip install -r ppocr_service/requirements.txt
```

سپس یکی از این دو راه:

```bash
# دستی:
conda activate ppocr_env
python ppocr_service/ppocr_server.py

# یا با دابل-کلیک روی ppocr.bat (ویندوز؛ خودش conda activate و اجرا را انجام می‌دهد)
```

سرویس روی `http://127.0.0.1:5005` بالا می‌آید و دو مسیر دارد: `GET /health` و `POST /recognize_batch`. با متغیر محیطی `PPOCR_USE_CPU=1` می‌توانید مجبورش کنید روی CPU اجرا شود، و با `PPOCR_REC_MODEL` مدل PaddleOCR دیگری (پیش‌فرض `PP-OCRv5_server_rec`) انتخاب کنید.

## نصب پیش‌نیازها

```bash
# داخل محیط اصلی (torch_env):
pip install -r requirements.txt
```

⚠️ برای `--detector east` باید فایل وزن `frozen_east_text_detection.pb` را دستی دانلود کرده و کنار `main.py` بگذارید (یا مسیرش را با `--east-model` بدهید) — وزن این مدل به‌صورت پکیج پایتون منتشر نشده است.

### اجرای آفلاین بعد از اولین دانلود مدل‌ها

مدل‌های Hugging Face (RT-DETRv2، TrOCR) و docTR بار اول به‌صورت خودکار دانلود می‌شوند. بعد از اولین اجرای موفق، برای جلوگیری از تلاش مجدد برای اتصال به اینترنت می‌توانید این متغیر محیطی را ست کنید:

```powershell
# PowerShell (ویندوز)
$env:HF_HUB_OFFLINE=1
```

```bash
# Linux / macOS
export HF_HUB_OFFLINE=1
```

## پیشنهادهای ترکیبی آماده

### real-time محض (دوربین زنده، سرعت اولویت)

```bash
python main.py --mode camera --detector dbnet --recognizer fast --fast-arch parseq
```

### تعادل خوب سرعت/دقت (نیاز به سرویس PP-OCRv5 روشن)

```bash
python main.py --mode file --file test.jpg --detector ensemble --recognizer ppocrv5
```

### بالاترین دقت واقع‌بینانه با سرعت هنوز قابل قبول (ترکیب پیشنهادی پروژه)

```bash
python main.py --mode file --file test.jpg --detector dbnet --recognizer ppocrv5
```

> ترکیب `dbnet` (تشخیص متن سریع و دقیق روی سند صاف‌شده) + `ppocrv5` (بازشناسی فوق‌سبک و دقیق) بهترین تعادل کلی پروژه بین سرعت و دقت است.

### بیشترین دقت ممکن با fallback هوشمند (برای صحنه‌های آزمایشگاهی که دنبال ۱۰۰٪ دقت هستند)

```bash
python main.py --mode file --file test.jpg --detector full --recognizer ppocrv5 --fallback-recognizer trocr-small --no-fp16
```

یا بدون وابستگی به سرویس خارجی:

```bash
python main.py --mode file --file test.jpg --detector full --recognizer trocr --trocr-size large --beams 4 --no-fp16
```

### ذخیره تصاویر مراحل میانی با حالت دیباگ

```bash
python main.py --mode file --file test.jpg --debug
```

### سبک‌ترین حالت برای سخت‌افزار ضعیف/CPU

```bash
python main.py --mode file --file test.jpg --detector dbnet --recognizer fast --fast-arch crnn_mobilenet_v3_small --no-fp16
```

### ذخیره‌ی کامل لاگ در فایل

```bash
python main.py --mode file --file sample.jpg --recognizer fast --fast-arch crnn_mobilenet_v3_large --log-level DEBUG > log.txt 2>&1
```

> **نکته:** مدل‌های StarNet-Tex و MTD (مقاله‌های ۲۰۲۵) بررسی شدند ولی چون وزن از‌پیش‌آموزش‌دیده‌ی رسمی/پکیج پایتونی عمومی برایشان پیدا نشد، در پروژه پیاده‌سازی نشدند. اگر لینک ریپوی رسمی یا فایل وزن مشخصی وجود دارد، می‌شود همان‌طور به لیست detectorها اضافه کرد.
