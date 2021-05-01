
$(document).ready(function(){

'use trick';


$("#btn-list").on("click",function(){
   $(".product-flower").addClass("pro-list");
    $("#btn-list").addClass("active-btn");
    $("#btn-grid").removeClass("active-btn");
});
$("#btn-grid").on("click",function(){
   $(".product-flower").removeClass("pro-list");
   $("#btn-list").removeClass("active-btn");
   $("#btn-grid").addClass("active-btn");
});
});