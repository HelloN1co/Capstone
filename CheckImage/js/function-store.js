$(document).ready(function(){
	'use trick'
	 // Tabs
    // $(".tab-content").hide();
    $("ul.tabs").each(function() {
        $(this).children().first().addClass("active");
        $(this).next().children().first().show().addClass("active");
    });
    $("ul.tabs li").each(function() {
        $(this).on("click", function() {
            var tab_content = $(this).parent().next().children();
            $(this).parent().children().removeClass("active");
            $(this).addClass("active");
            tab_content.hide().removeClass("active");
            var activeTab = $(this).attr("rel");
            $("#" + activeTab).fadeIn(400).addClass("active");
        });
    });
    // End tabs
});