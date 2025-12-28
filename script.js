const slider = document.getElementById('slider');
const foregroundImg = document.querySelector('.foreground-img');

slider.addEventListener('input', (e) => {
    const sliderValue = e.target.value;
    // Cập nhật clip-path để thay đổi phần ảnh hiển thị
    foregroundImg.style.clipPath = `inset(0 ${100 - sliderValue}% 0 0)`;
});
