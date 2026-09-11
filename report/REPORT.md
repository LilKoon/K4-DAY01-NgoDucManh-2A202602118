# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): (468, `cab`, 1, 0.510915, `ImageNet-1K`)
- Record này mô tả toàn ảnh như thế nào?: Đây là dự đoán cấp ảnh: checkpoint chọn `cab` là lớp có score cao nhất cho toàn bộ ảnh `traffic`, không phải một box hay một vật thể riêng lẻ. `rank=1` cho biết đây là lựa chọn đứng đầu trong top-5 của model.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?: Dataset/taxonomy mà checkpoint được huấn luyện trên đó định nghĩa class list; ở đây là `ImageNet-1K`. Checkpoint chỉ dự đoán trong danh sách này, không tự tạo thêm lớp theo guideline của dự án.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?: `class_id` là khóa ổn định để máy xử lý, `class_name` giúp con người đọc và kiểm tra, còn `taxonomy_name` cho biết ID/tên đó thuộc bộ lớp nào. Giữ cả ba giúp tránh nhầm lớp trùng tên hoặc ID giữa các taxonomy và giúp tái lập kết quả.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?: Cần quy định rõ mục tiêu của nhãn cấp ảnh và tiêu chí chọn một lớp đại diện, chẳng hạn chủ thể chính hoặc nội dung chi phối theo phạm vi dự án. Nếu không xác định được chủ thể chính, có nhiều chủ thể ngang nhau, hoặc không có lớp phù hợp trong taxonomy, annotator phải đánh dấu mơ hồ và chuyển reviewer/escalation thay vì tự chọn theo score.
- Vì sao model score không phải ground truth?: Model score chỉ phản ánh kết quả tính toán của checkpoint. Ground truth phải do con người xác nhận dựa trên taxonomy và guideline của dự án.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `person`, `0.912625`, `[385.33, 69.24, 498.92, 348.92]`, rộng `113.58` px và cao `279.68` px. Đây là prediction trong taxonomy `COCO-80`, với ngưỡng chạy `0.35`.
- Diễn giải vị trí box bằng lời: Hệ tọa độ là pixel, gốc ở góc trên bên trái ảnh `640 x 427`. Box bắt đầu khoảng `(x=385.33, y=69.24)` và kết thúc tại `(x=498.92, y=348.92)`, nên bao quanh người đứng ở phía bên phải ảnh, từ phần trên thân/người đến gần chân. `xyxy` là `[x_min, y_min, x_max, y_max]`, không phải `[x, y, width, height]`.
- So sánh số prediction ở hai threshold: Ở `0.20` có 17 vật thể; ở `0.35` có 11; ở `0.60` còn 6. Khi tăng threshold, các prediction có score thấp như `spoon`, `potted plant` và `bottle` bị loại trước; các lớp/đối tượng score cao vẫn còn.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp tăng độ bao phủ và cơ hội giữ vật thể nhỏ hoặc khó nhận diện, nhưng tạo thêm false positive và tăng số box reviewer phải kiểm tra. Threshold cao giảm khối lượng review và thường giữ prediction chắc hơn, nhưng có nguy cơ bỏ sót vật thể. Đây là tham số lọc prediction, không phải ngưỡng để quyết định ground truth.
- Đề xuất một quy tắc box chặt: Với mỗi object thuộc class trong guideline, vẽ một hình chữ nhật axis-aligned nhỏ nhất bao trọn phần object nhìn thấy, chạm sát biên ngoài của object và không lấy thêm nền hoặc object bên cạnh. Gán từng instance thành một box riêng; không dùng score của model để nới hoặc thu box.
- Với object bị che khuất/cắt mép, guideline cần quy định mức phần nhìn thấy tối thiểu, có gán nhãn khi chỉ nhận diện được một phần hay không, và box có bao phần bị che khuất hay chỉ phần nhìn thấy. Object bị cắt mép, che khuất nặng, hoặc không đủ thông tin để phân biệt class/instance phải được đánh dấu theo trạng thái quy định và chuyển reviewer/escalation; annotator không tự suy đoán phần không nhìn thấy.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `instance_id="kitchen-001"`, `class_name="person"`, score `0.899318`, có `348` điểm. Một phần polygon bắt đầu là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`; tọa độ dùng pixel trong ảnh `640 x 427`.
- Polygon bổ sung chi tiết gì so với box? Box chỉ cho phạm vi hình chữ nhật bao quanh object, nên có thể chứa nhiều nền. Polygon mô tả đường biên và hình dạng phần người được mask nhìn thấy; trong ảnh overlay, mask bám theo đầu, thân, tay và chân thay vì tô kín toàn bộ hình chữ nhật. Polygon vì vậy phù hợp hơn khi cần diện tích, biên hoặc phần object bị che bởi vật khác.
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` dùng để phân biệt từng object cụ thể trong output, liên kết class, score, box và polygon của cùng một instance, kể cả khi nhiều instance cùng class. Nó không phải `class_id`, không phải ID của taxonomy COCO-80, và cũng không phải track ID bảo đảm nhận diện cùng object qua nhiều frame/video.
- Đề xuất một quy tắc biên mask: Annotator vẽ mask sát biên phần object nhìn thấy, bao phủ các pixel thuộc object và không ăn vào nền hoặc object khác; giữ một polygon liên tục cho mỗi instance, thêm lỗ/đảo nếu công cụ và guideline hỗ trợ. Không dùng box hoặc score của model để tự động mở rộng mask; các chi tiết quá nhỏ so với độ phân giải cần theo tolerance được quy định trước.
- Với vùng mờ/tiếp xúc/che khuất, guideline cần quy định có mask phần bị che hay chỉ phần nhìn thấy, cách xử lý vùng xuyên qua/giữa các vật thể tiếp xúc, và ngưỡng kích thước hoặc độ rõ tối thiểu. Annotator chỉ áp dụng rule đã có; vùng biên không nhìn rõ, hai instance dính nhau, hoặc không xác định được phần thuộc object nào phải gắn cờ và chuyển reviewer/escalation thay vì đoán đường biên.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh theo taxonomy và guideline của dự án, ví dụ `class_id` + `class_name`. | Với `traffic`, model chọn `cab` score `0.510915`, có thể không đại diện cho mọi chủ thể trong ảnh; đây là prediction chứ chưa phải ground truth. | Xem toàn ảnh, áp dụng tiêu chí chọn nhãn cấp ảnh và ghi lớp đã được guideline xác nhận; nếu nhiều chủ thể ngang nhau hoặc không có lớp phù hợp thì đánh dấu mơ hồ và escalation. | Kiểm tra nhãn có đúng taxonomy, đúng phạm vi toàn ảnh và nhất quán với guideline; không dùng model score làm điểm chất lượng nhãn. |
| Phát hiện vật thể | Một class và một bounding box `xyxy` theo pixel cho mỗi instance được gán nhãn; các instance cùng class vẫn là các box riêng. | Ở `kitchen`, threshold `0.20` có thêm các prediction điểm thấp như `spoon`, `potted plant`, `bottle`; box nhỏ hoặc bị cắt mép dễ là false positive hoặc bị bỏ sót. | Gán từng object thuộc phạm vi, vẽ box chặt quanh phần nhìn thấy, không sao chép box/model score; đánh dấu ca che khuất, cắt mép hoặc không rõ class. | Kiểm tra bỏ sót, box có ăn nền/đè instance khác không, class có đúng không, và các prediction điểm thấp hoặc ca mơ hồ cần rework/escalation. |
| Instance segmentation | Một class, một `instance_id` và một polygon theo pixel cho mỗi instance; mask bám biên object theo guideline. | Overlay `kitchen` cho thấy nhiều object tiếp xúc/chồng gần nhau; biên mask quanh người, bàn, bát hoặc lò có thể khó tách chính xác khỏi nền/vật khác. | Vẽ polygon sát phần object nhìn thấy, giữ instance riêng, không đoán phần bị che; gắn cờ khi biên mờ hoặc không biết pixel thuộc instance nào. | Soi độ kín và độ sát của biên, vùng mask ăn vào nền hoặc object khác, instance trùng/chồng bất thường, và quyết định các ca che khuất/tiếp xúc. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng ba ảnh COCO công khai và các output đã được notebook tạo; không đưa ảnh cá nhân, dữ liệu nội bộ, thông tin nhận dạng, email hoặc số điện thoại vào Colab, báo cáo, JSON, PNG hay ZIP. Chỉ chia sẻ đúng repository/output cần nộp.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng, không tải xuống hoặc chia sẻ thêm, giữ nguyên evidence cần thiết để báo cáo sự cố, và báo cho Lab Coach/mentor hoặc kênh hỗ trợ chính thức của lớp.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
