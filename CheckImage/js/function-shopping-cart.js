$(document).ready(function(){
    "use strict";
if ($('.quantity').length) {
        var form_cart = $('form .quantity');
        form_cart.prepend('<span class="minus"><i class="fa fa-minus"></i></span>');
        form_cart.append('<span class="plus"><i class="fa fa-plus"></i></span>');

        var minus = form_cart.find($('.minus'));
        var plus = form_cart.find($('.plus'));

        minus.on('click', function() {
            var qty = $(this).parent().find('.qty');
            if (qty.val() <= 1) {
                qty.val(1);
            } else {
                qty.val((parseInt(qty.val(), 10) - 1));
            }

            var item_cart = $(this).parents('.item_cart');
            var item_number = item_cart.children('.product-quantity').find('input');
            var item_price = item_cart.children('.produc-price').children('input');
            var result =  item_cart.children('.total-price');
            result.html(function(){
                return '￥'+(item_number.val() * item_price.val().replace('￥', '')).toFixed(2);
            });

        });
        plus.on('click', function() {
            var qty = $(this).parent().find('.qty');
            qty.val((parseInt(qty.val(), 10) + 1));

            var item_cart = $(this).parents('.item_cart');
            var item_number = item_cart.children('.product-quantity').find('input');
            var item_price = item_cart.children('.produc-price').children('input');
            var result =  item_cart.children('.total-price');
            result.html(function(){
                return '￥'+(item_number.val() * item_price.val().replace('￥', '')).toFixed(2);
            });
        });
        $('.item_cart').each(function() {
        var answer = (parseInt($(this).children('.product-quantity').find('input').val(), 10) * $(this).children('.produc-price').find('input').val().replace('$', '')).toFixed(2);
        $(this).children('.total-price').html('￥' + answer);
        });

        $('.item_cart').each(function() {
        $(this).children('.product-quantity').find('input').change(function() {
          var answer = (parseInt($(this).val(), 10) * $(this).parents('.item_cart').children('.produc-price').find('input').val().replace('$', '')).toFixed(2);
          $(this).parents('.item_cart').children('.total-price').html('￥' + answer);
        });
        });
    }
});
