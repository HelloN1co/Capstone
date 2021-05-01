$(document).ready(function(){
  'use trick'
$('.btn-menu-mobile').on('click',function(){

if($(this).attr('data-click-state') == 1) {
$(this).attr('data-click-state', 0)
$(".menu-mobile").css("height","0").css("overflow","hidden");;
$(".icon-menu").css('display', 'inline-block')
$(".icon-bar-2").css('display', 'block')
$(".icon-bar-1").css('transform', 'unset')
$(".icon-bar-3").css('transform', 'unset')
} else {
$(this).attr('data-click-state', 1)
$(".menu-mobile").css("height","320px").css("overflow","auto");
$(".icon-menu").css('display', 'none')
$(".icon-bar-2").css('display', 'none')
$(".icon-bar-1").css('transform', 'rotate(45deg)  translate(5px)')
$(".icon-bar-3").css('transform', 'rotate(-45deg)  translate(5px)')
$(".icon-bar-1").css('transition', 'all .3s ease')
$(".icon-bar-3").css('transition', 'all .3s ease')
}

});
});