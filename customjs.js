const w = window.innerWidth;
const p = document.querySelector('p');

const getPercentX  = (x) => Math.round(x / w * 360);
const temp = document.getElementById('Home_main');
const styleEl = temp.style.background;
const setStyle = styleEl.setProperty.bind(styleEl);

document.addEventListener('mousemove', (e) => {
    const percentX  = getPercentX(e.clientX);
    const gradStart = `hsl(${percentX}, 100%, 75%)`;
    const gradEnd   = `hsl(${(percentX + 120) % 360}, 100%, 50%)`;

    setStyle('--grad-start', gradStart);
    setStyle('--grad-end', gradEnd);
    p.setAttribute('data-gradStart', gradStart);
    p.setAttribute('data-gradEnd', gradEnd);
});

$('.modal')
    .modal('show')
;