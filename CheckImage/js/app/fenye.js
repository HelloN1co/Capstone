define([
    'getEl',
    'data'
], function (getEl, data) {
    function Fenye(options) {
        Object.assign(this, {
            num: 5,
            abo: 0,
            index: 0,
        }, options);
        this.init();
    }

    Fenye.prototype.init = function () {
        this.randerOl(data.length);
        this.randerUl();
        this.olliClick();
        this.spanOnlick();
        this.fixdClick();
    }

    Fenye.prototype.fixdClick = function () {
        var str = "", strr = "";
        var that = this;
        // this.fixd.onclick = function () {
        //     var cont = "免费";
        //     [...data].filter(function (item) {
        //         if (item.mon.indexOf(cont) != -1) {
        //             str += `<li>
        //                     <img src="./img/${item.img}" alt="">
        //                     <h3>${item.name}</h3>
        //                     <span>${item.content}</span>
        //                     <p>${item.mon}</p>
        //                   </li>`
        //         }

        //     })
        //     that.ulf.innerHTML = str;
        //     str = "";
        // };
        // this.fixd2.onclick = function () {
        //     var cont = "收费";
        //     [...data].filter(function (item) {
        //         if (item.mon.indexOf(cont) != -1) {
        //             strr += `<li>
        //                     <img src="./img/${item.img}" alt="">
        //                     <h3>${item.name}</h3>
        //                     <span>${item.content}</span>
        //                     <p>${item.mon}</p>
        //                   </li>`
        //         }

        //     })
        //     that.ulf.innerHTML = strr;
        //     strr = "";
        // }

    }

    Fenye.prototype.spanOnlick = function () {
        var that = this;
        this.btom.addEventListener("click", function (e) {
            var tar = e.target;
            if (tar.className == "sy") {
                that.abo = 0;
                that.randerUl();
                if (that.olf.querySelector(".active")) {
                    that.olf.querySelector(".active").classList.remove("active");
                }
                that.olf.children[0].classList.add("active");
            }
            else if (tar.className == "wy") {
                that.abo = that.olf.children.length - 1;
                that.randerUl();
                if (that.olf.querySelector(".active")) {
                    that.olf.querySelector(".active").classList.remove("active");
                }
                that.olf.children[that.olf.children.length - 1].classList.add("active");
            }
            else if (tar.className == "up") {
                that.abo--;
                if (that.abo < 0) {
                    that.abo = that.olf.children.length - 1;
                }
                that.randerUl();
                if (that.olf.querySelector(".active")) {
                    that.olf.querySelector(".active").classList.remove("active");
                }
                that.olf.children[that.abo].classList.add("active");
            }
            else if (tar.className == "down") {
                that.abo++;
                if (that.abo > that.olf.children.length - 1) {
                    that.abo = 0;
                }
                that.randerUl();
                if (that.olf.querySelector(".active")) {
                    that.olf.querySelector(".active").classList.remove("active");
                }
                that.olf.children[that.abo].classList.add("active");
            }
        })
    }

    Fenye.prototype.olliClick = function () {
        var that = this;
        var olli = this.olf.children;
        [...olli].forEach(function (item, ind) {
            item.onclick = function () {
                that.abo = ind;
                if (that.olf.querySelector(".active")) {
                    that.olf.querySelector(".active").classList.remove("active");
                }
                item.classList.add("active");
                that.randerUl();
            }
        })
    }

    Fenye.prototype.randerOl = function (len) {
        // var olnum = len / this.num,
        //     str = "";
        // for (var i = 0; i < olnum; i++) {
        //     str += `<li ind="${i}" class="${i == 0 ? "active" : ""}">${i + 1}</li>`;
        // }
        // this.olf.innerHTML = str;
    }

    Fenye.prototype.randerUl = function () {
        // var begin = this.abo * this.num,
        //     over = (this.abo + 1) * this.num;
        // var dataslice = data.slice(begin, over),
        //     str = "";

        // str += dataslice.map(function (item) {
        //     return `<li>
        //                 <img src="./img/${item.img}" alt="">
        //                 <h3>${item.name}</h3>
        //                 <span>${item.content}</span>
        //                 <p>${item.mon}</p>
        //             </li>`
        // }).join("");
        //this.ulf.innerHTML = str;
    }
    
    return Fenye;
});