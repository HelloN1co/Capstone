$(document).ready(function(){

    var mapCanvas = document.getElementById("map");
    var mapOptions = {
        center: new google.maps.LatLng(40.753578, -73.982649),
        zoom: 18
    };
    var map = new google.maps.Map(mapCanvas, mapOptions);

});
