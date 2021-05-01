define(function() {
    var getEl={
        get:function(select,parent){
            var parentnode=parent||document;
            return typeof select==="string"?parentnode.querySelector(select):select;
        },
        gets:function(select,parent){
            var parentnode=parent||document;
            return typeof select==="string"?parentnode.querySelectorAll(select):select;
        }
    }
    return getEl;
});