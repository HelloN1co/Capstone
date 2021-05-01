$(document).ready(function(){
// function range prince-pc
$(function() {
 
            $( "#slider-3" ).slider({
               range:true,
               min: 150,
               max: 600,
               values: [ 150, 450 ],
               slide: function( event, ui ) {
                  $( "#price" ).val( "$" + ui.values[ 0 ] + "        -        $" + ui.values[ 1 ] );
               }
            });
            $( "#price" ).val( "$" + $( "#slider-3" ).slider( "values", 0 ) +
               "        -        $" + $( "#slider-3" ).slider( "values", 1 ) );
            
         });
// --------------------------------
// function range prince-mobile
$(function() {
  
            $( "#slider-3-mobile" ).slider({
               range:true,
               min: 150,
               max: 600,
               values: [ 150, 450 ],
               slide: function( event, ui ) {
                  $( "#price-mobile" ).val( "$" + ui.values[ 0 ] + " - $" + ui.values[ 1 ] );
               }
            });
            $( "#price-mobile" ).val( "$" + $( "#slider-3-mobile" ).slider( "values", 0 ) +
               " - $" + $( "#slider-3-mobile" ).slider( "values", 1 ) );
         });
// --------------------------------
});
