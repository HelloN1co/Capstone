$(document).ready(function(){
	'use trick'
	$("#input-search").on("click",function(){
        $('.content-search').css("visibility","visible").css("opacity","1");
    });
    $("#close-search").on("click",function(){
        $('.content-search').css("visibility","hidden").css("opacity","0");
    });
    $(".cart-menu").on("click",function(){
        $('.sidebar-cart').css("right","0").css("box-shadow","0 10px 20px 0");
    });
    $("#close-cart").on("click",function(){
         $('.sidebar-cart').css("right","-260px").css("box-shadow","unset");
    });
});