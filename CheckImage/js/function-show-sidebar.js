$(document).ready(function(){
	'use trick';
	$(".sp1").on("click",function(){
		$(".sidebar").addClass("sidebar-hide");
		$(".content-flower").addClass("content-flower-full");
		$(".product-flower").addClass("product-flower-col-3");
		$(".btn-hide").css("display","inline-block");
		$(".sp1").css("visibility","hiden").css("opacity","0");
	});
	$(".btn-hide").on("click",function(){
		$(".sidebar").removeClass("sidebar-hide");
		$(".content-flower").removeClass("content-flower-full");
		$(".product-flower").removeClass("product-flower-col-3");
		$(".btn-hide").css("display","none");
		$(".sp1").css("visibility","unset").css("opacity","1");
	});
});