require.config({
    paths:{
        fenye:"./app/fenye",
        getEl:"./conmon/getEl",
        data:"./mock/data",
    }
})

require(["fenye","getEl"],function(fenye,getEl){
    new fenye({
        box:getEl.get(".box"),// 上部
        ulf:getEl.get(".ulf"),//渲染内容区域
        btom:getEl.get(".btom"),//下部
        olf:getEl.get(".olf"),//下方分页器的区域
        // fixd:getEl.get(".fixd"),//免费按钮筛选
        // fixd2:getEl.get(".fixd2"),//收费按钮筛选
    })
})