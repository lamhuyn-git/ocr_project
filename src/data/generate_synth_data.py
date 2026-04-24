"""
Sinh ảnh text tổng hợp cho 69 ký tự tiếng Việt còn thiếu trong dataset.
Output: image_train/synth/crop_img/ + rec_gt_synth.txt (PaddleOCR format)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.synth_augment import render_variants

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 69 ký tự còn thiếu
MISSING_CHARS = set("ÀÁÂÈÉÌÍÓÕÙÝĂƠẠẢẤẦẨẪẬẮắẰằẲẳẴẵẶẸẺẻẼẽẾếỀỂểỄỈỉỊỌỎỏỐỒổỖỚỞỠỡỢỤụỨỪừỬỮỰỲỴỵỶỷỸ")

# Phrases thực tế từ form CT01 và hành chính Việt Nam chứa các ký tự thiếu
PHRASES = [
    # Tên người (chứa À,Á,Â,È,É,Ì,Í,Ó,Ù,Ý,Ă,Ơ)
    "Nguyễn Thị Bào", "Lê Văn Ừa", "Trần Thị Ắng", "Phạm Văn Ổn",
    "Hoàng Thị Ụy", "Đỗ Văn Ởng", "Vũ Thị Ợi", "Bùi Văn Ẫu",
    "Phan Thị Ẻo", "Đinh Văn Ẽo", "Lý Thị Ẹo", "Mai Văn Ẳng",
    "Cao Thị Ặc", "Dương Văn Ẵng", "Tô Thị Ẩu", "Hà Văn Ầu",
    "Đặng Thị Ấu", "Võ Văn Ẫu", "Trương Thị Ậu", "Ngô Văn Ắc",
    # Địa chỉ (chứa Ấ,Ầ,Ổ,Ộ,Ờ,Ở,Ợ,Ụ,Ừ,Ự)
    "Ấp Bình An, xã Phú Hội",
    "Ầu Một, phường Bình Thọ",
    "Ổ Gà, xã Tân Thới Hiệp",
    "Phường Ô Môn, quận Bình Thủy",
    "Ừ thì tại số 12 đường Ắc Quy",
    "Xã Ổn Định, huyện Càng Long",
    "Thị trấn Ớt, tỉnh Bình Dương",
    "Đường Ờ La, phường Phú Mỹ",
    "Số 45 Ỡm La, quận 12",
    "Hẻm 3 Ợi Lợi, phường Tân Phú",
    # Nội dung form CT01 (chứa Ề,Ể,Ễ,Ệ,Ỉ,Ị,Ọ,Ỏ,Ố,Ồ)
    "Đề nghị đăng ký tạm trú tại địa chỉ trên",
    "Ề nghị xác nhận thông tin cư trú",
    "Ổn định chỗ ở tại phường Phú Hữu",
    "Đã thuê nhà tại số Ổn định",
    "Ỏi thật sự tại địa chỉ đã khai",
    "Ố số nhân khẩu trong hộ gia đình",
    "Ồi lại tại địa chỉ thường trú cũ",
    "Ổi theo hộ khẩu số 186 đường Bưng Ông Thoàn",
    # Ngày tháng (chứa Ộ,Ớ,Ờ)
    "Ngày 17 tháng 11 năm 2023",
    "Ngày 01 tháng 10 năm 2024",
    "Lập ngày 05 tháng 03 năm 2025",
    "Có hiệu lực từ ngày Ổn định",
    # Chức vụ / cơ quan (chứa Ụ,Ủ,Ứ,Ừ,Ử,Ữ,Ự)
    "Ủy ban nhân dân phường Phú Hữu",
    "Công an phường Ổn Định",
    "Trưởng Công an phường xác nhận",
    "Cán bộ tiếp nhận hồ sơ",
    "Ủy ban Nhân dân Thành phố Hồ Chí Minh",
    "Ừ thì đây là xác nhận chính thức",
    # Từ viết hoa thường gặp (Ỳ,Ỵ,Ỷ,Ỹ,Ỳ)
    "Ỳ La thôn Ỵ An",
    "Hội Ỷ Lan phường Ỳ Đan",
    "Khu vực Ỹ Lan, tỉnh Hưng Yên",
    "TỜ KHAI THAY ĐỔI THÔNG TIN CƯ TRÚ",
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
    "PHIẾU BÁO THAY ĐỔI HỘ KHẨU",
    # Cụm từ ngắn chứa ký tự hiếm
    "Ắc quy điện", "Ằng hắng ho", "Ẳng đặng", "Ẵng mèo kêu",
    "Ặc nước", "Ẹc miệng", "Ẻo léo người", "Ẽo éo", "Ẹo oặt",
    "Ếch nhái", "Ềm ếm", "Ểm đềm", "Ễnh bụng", "Ỉu xỉu",
    "Ịu người", "Ọc ạch", "Ỏng ẻo", "Ốc sên", "Ồm ộp",
    "Ổ bánh mì", "Ỗn ào", "Ộc lốc", "Ớt cay", "Ờ hờ",
    "Ởn gáy", "Ỡm lợm", "Ợ hơi", "Ụt ịt", "Ủ rũ",
    "Ứa nước mắt", "Ừ thôi", "Ửa nước", "Ữa chữa", "Ựa mạnh",
    "Ỳ hạch", "Ỵ tủy", "Ỷ lại", "Ỹ thiện",
    # Tên tỉnh thành có ký tự thiếu
    "Tỉnh Ắc Lắk", "Huyện Ấm Thượng", "Xã Ổn Điền",
    "Tỉnh Bà Rịa Vũng Tàu", "Thành phố Bảo Lộc",
    "Huyện Bảo Lâm tỉnh Lâm Đồng",
    # Bổ sung cho ký tự còn thiếu: À Â È É Ì Í Ó Õ Ù Ý Ă Ơ Ả ằ ẳ ẵ ẽ ể ỏ ổ ỡ ỵ ỷ
    "Àng trời tối", "Bà Àng sống tại đây",
    "Âm mưu phá hoại", "Ân nhân cứu giúp", "Âu lo buồn bã",
    "Èo éo tiếng kêu", "Ẽo éo mèo kêu",
    "Éo le hoàn cảnh", "Bé Éo tên thật",
    "Ìm lặng không nói", "Ít ỏi tài sản",
    "Ích lợi cho cộng đồng", "Ía ra ngoài",
    "Óc não thông minh", "Óc chó nhân",
    "Õng ẹo đi lại",
    "Ùa vào trong nhà", "Ùn ùn kéo đến",
    "Ý kiến đóng góp xây dựng", "Ý chí vươn lên",
    "Ăn uống sinh hoạt bình thường", "Ăn no mặc ấm",
    "Ơn nghĩa sinh thành", "Ơn trời mưa nắng",
    "Ảnh chụp căn cước công dân", "Ảnh hưởng đến sinh hoạt",
    "bằng lòng với kết quả", "ngang bằng nhau",
    "ẳng thẳng ngang hàng", "ngã ẳng xuống đất",
    "mèo ẵng kêu lên", "con ẵng nằm im",
    "tiếng ẽo éo vang lên", "màu ẽo nhạt dần",
    "để thể hiện rõ ràng", "thể ể mà không làm",
    "tỏ ỏ ra vui mừng", "nhỏ ỏ tiếng nói",
    "ổn định cuộc sống", "tổ chức buổi lễ",
    "lỡ ỡ tay đánh rơi", "ngỡ ỡ ngàng trước cảnh",
    "tủy ỵ sống nuôi cơ thể", "cốt tủy ỵ vấn đề",
    "ỷ lại vào người khác", "ỷ thế cậy quyền",
]


def get_label_path(output_dir: str) -> str:
    return os.path.join(output_dir, "rec_gt_synth.txt")


def filter_phrases_with_missing_chars(phrases: list) -> list:
    """Chỉ giữ phrase có chứa ít nhất 1 ký tự còn thiếu."""
    return [p for p in phrases if any(c in MISSING_CHARS for c in p)]


def main():
    output_dir = os.path.join(PROJECT_ROOT, "image_train", "synth")
    img_dir    = os.path.join(output_dir, "crop_img")
    os.makedirs(img_dir, exist_ok=True)

    fonts = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans-Bold.ttf",
    ]
    fonts = [f for f in fonts if os.path.exists(f)]
    if not fonts:
        print("ERROR: Không tìm thấy font hỗ trợ tiếng Việt.")
        return

    valid_phrases = filter_phrases_with_missing_chars(PHRASES)
    print(f"Phrases hợp lệ: {len(valid_phrases)}")

    label_lines = []
    total = 0

    for idx, text in enumerate(valid_phrases):
        variants = render_variants(text, fonts, img_dir, prefix=f"synth_{idx:04d}")
        for fname in variants:
            rel_path = os.path.join("crop_img", fname)
            label_lines.append(f"{rel_path}\t{text}")
            total += 1

    label_path = get_label_path(output_dir)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines) + "\n")

    print(f"Đã gen {total} ảnh từ {len(valid_phrases)} phrases")
    print(f"Label file: {label_path}")
    print(f"Ảnh tại: {img_dir}")

    # Kiểm tra coverage các ký tự còn thiếu
    covered = set()
    for text in valid_phrases:
        covered.update(c for c in text if c in MISSING_CHARS)
    still_missing = MISSING_CHARS - covered
    print(f"\nKý tự đã cover: {len(covered)}/{len(MISSING_CHARS)}")
    if still_missing:
        print(f"Vẫn còn thiếu: {''.join(sorted(still_missing))}")


if __name__ == "__main__":
    main()
