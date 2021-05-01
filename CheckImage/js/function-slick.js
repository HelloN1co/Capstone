$(document).ready(function(){


// ---------slick simple-------------
var x= $(window).width();
if(x>1200){
  $('.product-slick').slick({
  infinite: true,
  slidesToShow: 4,
  slidesToScroll: 4,

});
};
if(x>992 && x<1200){
  $('.product-slick').slick({
  infinite: true,
  slidesToShow: 3,
  slidesToScroll: 3,

});
  };
  if(x>567 && x<992){
  $('.product-slick').slick({
  infinite: true,
  slidesToShow: 2,
  slidesToScroll: 2,

});
  };
  if(x<567){
  $('.product-slick').slick({
  infinite: true,
  slidesToShow: 1,
  slidesToScroll: 1,

});
};

// ---------slick syncing-------------
$('.slider-for').slick({
  slidesToShow: 1,
  slidesToScroll: 1,
  arrows: false,
  fade: true,
  asNavFor: '.slider-nav'
});
$('.slider-nav').slick({
  slidesToShow: 3,
  slidesToScroll: 1,
  asNavFor: '.slider-for',
  centerMode: true,
  focusOnSelect: true
});
});