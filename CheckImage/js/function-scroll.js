  $(document).ready(function() {
      $('body').css('visibility','visible');
      // TimelineLite for title animation, then start up superscrollorama when complete
      (new TimelineLite({onComplete:initScrollAnimations}))       
        .from( $('#title-line2'), .4, {css:{top:'1000px',opacity:'0'}, ease:Expo.easeOut})  
      function initScrollAnimations() {
        var controller = $.superscrollorama();
        // showcase tweens
        controller.addTween('#showcase h1', TweenMax.from( $('#showcase h1'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
         controller.addTween('#showcase h2', TweenMax.from( $('#showcase h2'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
        controller.addTween('#showcase p', TweenMax.from( $('#showcase p'), 1, {css:{opacity:0}, ease:Quad.easeOut}));
        $('#showcase .gallery figure').css('position','relative').each(function() {
          controller.addTween('#showcase .gallery', TweenMax.from( $(this), 1, {delay:Math.random()*.2,css:{left:0,top:Math.random()*200-100,opacity:0},
           ease:Back.easeOut}));
        });
        // showcase tweens 2
        controller.addTween('#showcase-2 h1', TweenMax.from( $('#showcase-2 h1'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
         controller.addTween('#showcase-2 h2', TweenMax.from( $('#showcase-2 h2'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
        controller.addTween('#showcase-2 p', TweenMax.from( $('#showcase-2 p'), 1, {css:{opacity:0}, ease:Quad.easeOut}));
        $('#showcase-2 .gallery figure').css('position','relative').each(function() {
          controller.addTween('#showcase-2 .gallery', TweenMax.from( $(this), 1, {delay:Math.random()*.2,css:{left:0,top:Math.random()*200-100,opacity:0},
           ease:Back.easeOut}));
        });
        // showcase tweens 3
        controller.addTween('#showcase-3 h1', TweenMax.from( $('#showcase-3 h1'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
         controller.addTween('#showcase-3 h2', TweenMax.from( $('#showcase-3 h2'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
        controller.addTween('#showcase-3 p', TweenMax.from( $('#showcase-3 p'), 1, {css:{opacity:0}, ease:Quad.easeOut}));
        $('#showcase-3 .gallery figure').css('position','relative').each(function() {
          controller.addTween('#showcase-3 .gallery', TweenMax.from( $(this), 1, {delay:Math.random()*.2,css:{left:0,top:Math.random()*200-100,opacity:0},
           ease:Back.easeOut}));
        });
        // showcase tweens 4
        controller.addTween('#showcase-4 h1', TweenMax.from( $('#showcase-4 h1'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
         controller.addTween('#showcase-4 h2', TweenMax.from( $('#showcase-4 h2'), .75, {css:{letterSpacing:20,opacity:0}, ease:Quad.easeOut}));
        controller.addTween('#showcase-4 p', TweenMax.from( $('#showcase-4 p'), 1, {css:{opacity:0}, ease:Quad.easeOut}));
        $('#showcase-4 .gallery figure').css('position','relative').each(function() {
          controller.addTween('#showcase-4 .gallery', TweenMax.from( $(this), 1, {delay:Math.random()*.2,css:{left:0,top:Math.random()*200-100,opacity:0},
           ease:Back.easeOut}));
        });
      }
    });