$(document).ready(function(){
'use trick'
	$(window).scroll(function(){ 
        if ($(this).scrollTop() > 50) { 
            $('.back-to-top').removeClass("fade");
            $('.back-to-top').fadeIn(); 
        } else { 
            $('.back-to-top').fadeOut(); 
        } 
    }); 
    $('.back-to-top').on("click",function(){ 
        $("html, body").animate({ scrollTop: 0 }, 500); 
        return false; 
    }); 

});