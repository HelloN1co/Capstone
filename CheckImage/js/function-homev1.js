
$(document).ready(function(){
'use trick';
$("body,html").on("loaded",function(){
	$("#header-v1").addClass("pos-header");
}); 
// ---------scroll-------------------
window.onscroll = function() {scrollFunction()};
function scrollFunction() {
 
  if ($("body,html").scrollTop() > 50) {
  	$("#header-v1").css("transition","all .5s ease");
    $("#header-v1").css("background","rgba(255,255,255,0.95)");
    $("#header-v1").css("box-shadow","0 10px 10px rgba(0,0,0,0.05)");
    $("#header-v1").removeClass("pos-header");
    $("nav").css("display","none");
  } else {
  		$("#header-v1").addClass("pos-header");
     	$("#header-v1").css("background","unset");
    	$("nav").css("display","block");
    	$("#header-v1").css("box-shadow","unset");
    
  }

};



// ----------------------------------------------------------------
// -------time counter-------------------
var target_date = new Date().getTime() + (76000*3600*48); // set the countdown date
var days, hours, minutes, seconds; // variables for time units

var countdown = document.getElementById("tiles"); // get tag element

getCountdown();

setInterval(function () { getCountdown(); }, 1000);

function getCountdown(){

	// find the amount of "seconds" between now and target
	var current_date = new Date().getTime();
	var seconds_left = (target_date - current_date) / 1000;

	days = pad( parseInt(seconds_left / 86400) );
	seconds_left = seconds_left % 86400;
		 
	hours = pad( parseInt(seconds_left / 3600) );
	seconds_left = seconds_left % 3600;
		  
	minutes = pad( parseInt(seconds_left / 60) );
	seconds = pad( parseInt( seconds_left % 60 ) );

	// format countdown string + set tag value
	countdown.innerHTML = "<span>" + days + "</span><span>" + hours + "</span><span>" + minutes + "</span><span>" + seconds + "</span>"; 
}

function pad(n) {
	return (n < 10 ? '0' : '') + n;
}
// ---------------------------------------------------------------

});
