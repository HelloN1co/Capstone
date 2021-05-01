
function popup(){
    $(".modal").addClass("in");
    $(".modal").css("display","block");
    $(".modal").css("background-color","rgba(100, 100, 98, .9)");
};
$(".close").on("click",function(){
     $(".modal").css("display","none");
});
$(document).ready(function(){
    'use trick'
	 $(function(){
        var $header = $('.carousel');
        var $window = $(window).on('resize', function(){
           var height = $(this).height();
           $header.height(height);
          
        }).trigger('resize'); //on page load

    });
});

