$(document).ready(function(){
	'use trick'
	$(".mobile-menu").on("click",function(){
		$("#myNavbar").css("left","0").css("overflow-y","auto");
		$(".BG-menu").addClass("active-bg-menu");
	});
	$("#btn-close-menu").on("click",function(){
		$("#myNavbar").css("left","-260px");
		$(".BG-menu").removeClass("active-bg-menu");
	});


	//------------ sidebar-------------------- 
	$(".mobile-sidebar").on("click",function(){

		if($(this).attr('data-click-state') == 1) {
			$(this).attr('data-click-state', 0)
			$(".mobile-sidebar").css("transform","translateX(0)");
			$("#icon-sidebar").css("transform","rotateY(0)");
			$("#mysidebar").css("right","-260px").css("overflow","hidden");
			$(".btn-mobile-sidebar").css("background","#f3f3f5").css("box-shadow","0 10px 20px 0 rgba(0,0,0,.1)");
			$(".BG-menu").removeClass("active-bg-menu");
		} else {
			$(this).attr('data-click-state', 1)
			$(".BG-menu").addClass("active-bg-menu");
			$(".mobile-sidebar").css("transform","translateX(-225px)");
			$(".btn-mobile-sidebar").css("background","#fff").css("box-shadow","none");
			$("#mysidebar").css("right","0").css("overflow-y","auto");
			$("#icon-sidebar").css("transform","rotateY(180deg)");
			}
	});
	
});