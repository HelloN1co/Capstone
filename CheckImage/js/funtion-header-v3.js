$(document).ready(function(){
'use trick';

// ---------scroll-------------------
window.onscroll = function() {scrollFunction()};
function scrollFunction() {
    var x = $(window).width();
  if ($("body,html").scrollTop() > 50 && x <=812) {
  	$("#header-v3").addClass("navbar-fixed-top").css("background","rgba(255,255,255,.8)").css("width","100%");
    
  } else {
  		$("#header-v3").removeClass("navbar-fixed-top").css("width","");;
  }

};
});