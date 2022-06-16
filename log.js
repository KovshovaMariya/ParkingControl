let DBLength = 0;

const getDBLength = (db) => {
    return db.exec("SELECT * FROM LOG")[0].values.length;
}

const mapValuesToHTML = (values, last) => {
    if (last === 0)
        values.map((value) => {
            let div = document.createElement("div");
            div.innerHTML = value[0] + ' ' + value[1].substring(3, 20) + value[2];
            document.getElementById("main").appendChild(div);
        })
    else {
        let tempVal = values[last - 1];
        let div = document.createElement("div");
        div.innerHTML = tempVal[0] + ' '  + tempVal[1].substring(3, 20) + tempVal[2];
        document.getElementById("main").appendChild(div);
    }
}

const worker = () => {
    let tempDBLength = 0;
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/parking-log.db', true);
    xhr.responseType = 'arraybuffer';

    xhr.onload = function (e) {
        var uInt8Array = new Uint8Array(this.response);
        var db = new SQL.Database(uInt8Array);
        var contents = db.exec("SELECT * FROM LOG");
        values = contents[0].values;
        DBLength = values.length;
        console.log(values)
        mapValuesToHTML(values, 0);
    };
    xhr.send();


    let uInt8Array;
    let db;
    let contents;
    setInterval(() => {
        xhr.open('GET', '/parking-log.db', true);
        xhr.responseType = 'arraybuffer';

        xhr.onload = function (e) {
            uInt8Array = new Uint8Array(this.response);
            db = new SQL.Database(uInt8Array);
            contents = db.exec("SELECT * FROM LOG");
            tempDBLength = contents[0].values.length;
            console.log(tempDBLength)
            if (DBLength < tempDBLength) {
                let values = contents[0].values;
                console.log(values)
                mapValuesToHTML(values, tempDBLength);
                DBLength = tempDBLength
            }
        };
        xhr.send();

    }, 5000)
}

worker();