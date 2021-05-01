/*
*上传图片并本地预览插件
*兼容IE8
*obj    ----上传图片容器
*picNum ----上传图片的张数
*width  ----图片宽度
*height ----图片宽度
 */
(function($) {
	// 构造函数
	function UploadImg(obj,opt) {
		this.obj = obj;
		this.$obj = $(obj);
		if(opt.picStyle==0){
			this.defaultOpt={
				"picStyle":0,
				"picNum":1,
				"width":200,
				"height":200
			};
			this.options = $.extend({},this.defaultOpt,opt);
		}
		if(opt.picStyle==1){
			//如果用户选择了自定义样式，但又没有选择宽度及高度，则会采用默认宽高
			this.defaultOpt = {
				"picStyle":1,
				"picNum": 1,
				"width": 200,
				"height": 200,
				"realWidth":200,
				"realHeight":200
			};
			// this.num = 0;//计算上传图片个数
			this.options = $.extend({},this.defaultOpt,opt);
		}
		console.log(this.options);
		this.init();
	};
// 初始化html
UploadImg.prototype.init = function() {
	// var html = '<input type="file" class="fileInput" id="file" >'
	// 			+'<div class="previewBox" id="previewBox" style="position: absolute; z-index:-1; width: '+this.options.maxwidth+'; height: '+this.options.maxheight+';" ><img id="uploadImg" src="uploadImg.jpg"  width='+ this.options.width +' height='+ this.options.height +'></div>';
	var html = '<input type="file" name="multipartFile" class="fileInput" id="file" >'
				+'<div class="previewBox" id="previewBox" style="position: absolute; display: inline-block; z-index:-1;max-width: '+this.options.maxwidth+'; max-height: '+this.options.maxheight+';" ><img id="uploadImg" src="img/uploadImg.png"  width='+ this.options.width +' height='+ this.options.height +'>' ;
				// +'<div class="picBox" id="picBox"><img id="uploadImg" src="uploadImg.png"></div>';
	this.$obj.append($(html));
	this.bindEvent();
}
//绑定事件
UploadImg.prototype.bindEvent = function() {
	var self = this;
	this.$obj.on("click.choose",".previewBox",function() {
		$(".fileInput").trigger("click");
	});
	$(".fileInput").on("change.upload",function() {
		self.operationImg(this);
	});
	this.$obj.off(".choose,.upload");
}
//检查图片宽高
UploadImg.prototype.testWidthHeight = function(width,height){
	debugger;
	//如果picStyle是0，则只能按照指定大小进行裁剪图片
	if(this.options.picStyle==0){
		if(width===this.options.width &&height===this.options.height){
			alert("The selected picture is just right");
			return true;
		}else{
			alert("The width and height of the picture you selected does not meet the requirements. Please select the width"+this.options.width+"px and the height is"+this.options.height+"px");
			return  false;
		}
	}
	if(this.options.picStyle==1){
		//计算上传图片与原本图片框的比例
		debugger;
		var heightScale=height/this.options.height;
		var widthScale=width/this.options.width;
		console.log(heightScale);
		console.log(widthScale);
		if(heightScale>widthScale){
			this.options.realWidth=width/heightScale;
			this.options.realHeight=this.options.height;
		}else{
			this.options.realHeight=height/widthScale;
			this.options.realWidth=this.options.width;
		}
		return true;
	}
}

UploadImg.prototype.testWidthHeightIE = function(DomElement){
	var img = new Image();
	img.src = document.getElementById('file').value;
	width=img.width;
	height=img.height;
	console.log(img.width+img.height);
}
//检查图片格式
UploadImg.prototype.isImg = function(url) {
	var result = /.+\.(jpg|png|jpeg|gif)$/.test(url);
	if(!result) {
		alert("The format of the picture you selected is incorrect. Please select again");
		return false;
	} else {
		return true;
	}
},
//添加预览图片到页面上
UploadImg.prototype.addImgHtml = function(url) {
	//可以上传多张图片
	// if(this.options.picNum == 1) {
	// 	if($(".previewBox")) {
	// 		$(".previewBox").html("<img src="+ url +" width="+ this.options.width +" height="+ this.options.height +">");
	// 	}
	// } else {
	// 	if($(".previewBox") && this.num < this.options.picNum) {
	// 		$(".previewBox").append("<img src="+ url +" width="+ this.options.width +" height="+ this.options.height +">");
	// 		this.num++;
	// 	}
	// }
	if($(".previewBox")) {
		$(".previewBox").html("<img src="+ url +" width="+ this.options.realWidth +" height="+ this.options.realHeight +">");
	}
},
//兼容IE处理 填充
UploadImg.prototype.previewImgIE = function(obj) {
	obj.select();
	$(obj).blur();
	if(document.selection) {
		var url = document.selection.createRange().text;
		if(this.isImg(url)) {
			var imgWrap = "<div class='imgWrap' id='img'></div>";
			if(this.options.picNum == 1) {
				$(".previewBox").html($(imgWrap));
			} else if(this.options.picNum > 1 && this.num < this.options.picNum) {
				$(".previewBox").append($(imgWrap));
				this.num++;
			} else {
				return;
			};
				$(".imgWrap").css({
					"width":this.options.width,
					"height":this.options.height,
					"display":"inline-block",
					"margin-right":"10px",
					"*display":"inline",
					"*zoom":1
				});
			$(".imgWrap:last").css("filter","progid:DXImageTransform.Microsoft.AlphaImageLoader(sizingMethod = scale,src=\""+url+"\")");
			this.testWidthHeightIE(document.getElementById("file"));
		}
	}
},
//正常处理 不填充
UploadImg.prototype.previewImg = function(obj) {
	debugger;
	var file = obj.files[0];
	var self = this;
	if(this.isImg(file.name)) {
		var reader = new FileReader();
		reader.onload = function(evt) {
			var data=evt.target.result;
			var image =new Image();
			image.onload=function(){

				if(self.testWidthHeight(image.width,image.height)){
					self.addImgHtml(evt.target.result);
					$("#uploadImg").css("display", "block");
				}
			}
			image.src=data;
		};
		reader.readAsDataURL(file);

	} else {
		alert("The format of the picture you entered is incorrect. Please re-enter it");
		return false;
	}
};
//上传图片操作；
UploadImg.prototype.operationImg = function(fileObj) {
	if(fileObj.files && fileObj.files[0]) {
		//html5 files API
		this.previewImg(fileObj);
	} else {
		//兼容IE
		this.previewImgIE(fileObj);
	}
}
//绑定插件
$.fn.uploadImg= function(options) {
	return this.each(function() {
		new UploadImg(this,options);
	});
}
})(jQuery);