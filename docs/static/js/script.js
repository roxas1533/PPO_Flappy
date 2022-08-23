const HEIGHT = 400;
const WIDTH = 400;
class Vector {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

class GameObject {
    constructor(x, y, width, height, tag, image = null) {
        this.pos = new Vector(x, y);
        this.velocity = new Vector(0, 0);
        this.size = new Vector(width, height);
        if (image) {
            this.image = image;
        }
        this.isDeath = false;
        this.tag = tag;
    }
    draw(ctx) {
        if (this.image) {
            if (this.tag === "player") {
                ctx.drawImage(
                    this.image,
                    this.pos.x,
                    this.pos.y,
                    this.size.x,
                    this.size.y
                );
            } else {
                ctx.save();

                if (this.tag == "UP") {
                    ctx.scale(1, -1);
                }
                ctx.drawImage(
                    this.image,
                    0,
                    0,
                    this.image.width,
                    this.size.y,
                    this.pos.x,
                    this.pos.y - (this.tag == "UP" ? this.size.y : 0),
                    this.image.width,
                    this.size.y
                );
                ctx.restore();
            }
        }
    }
    collides(other) {
        return (
            this.pos.x < other.pos.x + other.size.x &&
            this.pos.x + this.size.x > other.pos.x &&
            this.pos.y < other.pos.y + other.size.y &&
            this.pos.y + this.size.y > other.pos.y
        );
    }
}
function make_image(img) {
    const _img = new Image();
    _img.src = img;
    return _img;
}

class Obstacle extends GameObject {
    static pipe_image = make_image("static/images/pipe-green.png");
    constructor(x, y, width, height, tag) {
        super(x, y, width, height, tag, Obstacle.pipe_image);
        this.velocity = new Vector(-4, 0);
    }
    update() {
        this.pos.x += this.velocity.x;
        if (this.pos.x + this.width < 0) {
            this.isDeath = true;
        }
    }
}
class Point extends GameObject {
    constructor(x, y, width, height, tag) {
        super(x, y, width, height, tag);
        this.velocity = new Vector(-4, 0);
    }
    update() {
        this.pos.x += this.velocity.x;
        if (this.pos.x + this.width < 0) {
            this.isDeath = true;
        }
    }
}

class Player extends GameObject {
    #G = 1.0;
    static player_image = make_image("static/images/Flappy.png");
    constructor() {
        super(100, HEIGHT / 2, 25, 25, "player", Player.player_image);
    }
    update() {
        this.velocity.y += this.#G;
        this.velocity.y = Math.max(this.velocity.y, -8);
        this.velocity.y = Math.min(this.velocity.y, 10);
        this.pos.x += this.velocity.x;
        this.pos.y += this.velocity.y;
        if (this.pos.y < 0 || this.pos.y + this.size.y > HEIGHT) {
            this.isDeath = true;
        }
    }
    jump() {
        this.velocity.y = -9;
    }
    reset() {
        this.pos.x = 100;
        this.pos.y = HEIGHT / 2;
        this.velocity = new Vector(0, 0);
        this.isDeath = false;
    }
}

class GameScene {
    constructor() {
        this.high_score_area = document.getElementById("high_score");
        this.scene_type = ["start", "game"];
        this.scene = "start";
        this.Player = new Player();
        document.addEventListener("keydown", this.keyHandler.bind(this));
        this.obstacles = [];
        this.time = 0;
        this.score = 0;
        this.high_score = 0;
        this.input_data = [];
        this.model = new PPO();
    }
    async render(ctx, canvas) {
        this.obstacles.forEach((e) => {
            e.draw(ctx, "obstacle");
        });
        this.Player.draw(ctx);

        if (this.scene === "game") {
            // const ctx2 = document.getElementById("aaaa").getContext("2d");
            const resized = resize(canvas);
            const gray_data = gray(ctx.getImageData(0, 0, 400, 400));
            // ctx2.putImageData(gray_data, 0, 0);

            // const gray_data = Array(256 * 256).fill(1);
            this.input_data = stack_frame(gray_data, this.input_data);
            const aa = document.getElementById("aa");
            if (this.time == 2) {
                let t = "";
                // for (let i = 0; i < this.input_data.length; i += 5) {
                //     const ii = i + 4;
                //     t += this.input_data[ii] + ",";
                //     if ((i / 5 + 1) % 400 === 0 && i != 0) {
                //         t += "\n";
                //     }
                // }
                for (let i = 0; i < gray_data.length; i++) {
                    t += gray_data[i] + ",";
                    if ((i + 1) % 400 === 0 && i != 0) {
                        t += "\n";
                    }
                }
                aa.innerHTML = t;
            }
            const data = Float32Array.from(this.input_data);
            const data_tensor = new ort.Tensor(
                "float32",
                data,
                [1, 5, 256, 256]
            );
            // if (this.model.loaded) {
            //     const act = await this.model.predict(data_tensor);

            //     console.log(act);
            //     if ((Math.random() < act[0] ? 0 : 1) == 1) {
            //         this.Player.jump();
            //     }
            // }
        }
        // let sum = 0;
        // for (let i = 0; i < this.input_data.length; i++) {
        //     sum += this.input_data[i];
        // }
        // console.log(sum / this.input_data.length);
        ctx.font = "30px Brush Script MT";
        ctx.fillStyle = "white";
        const text = "SCORE: " + this.score;
        ctx.fillText(text, 200 - ctx.measureText(text).width / 2, 30);
    }
    async update() {
        if (this.scene === "game") {
            this.Player.update();
            this.obstacles.forEach((e) => {
                e.update();
                if (e.collides(this.Player)) {
                    if (e.tag === "player") {
                        e.isDeath = true;
                        this.score += 1;
                    } else {
                        this.Player.isDeath = true;
                    }
                }
            });
            this.obstacles = this.obstacles.filter((e) => {
                return !e.isDeath;
            });
            if (this.time % 50 == 0) {
                const rand = Math.random() * (400 - 150);
                this.obstacles.push(new Obstacle(WIDTH, 0, 52, rand, "UP"));
                this.obstacles.push(
                    new Obstacle(WIDTH, rand + 120, 52, HEIGHT, "DOWN")
                );
                this.obstacles.push(
                    new Point(WIDTH + 25 + 15, rand, 5, 120, "player")
                );
            }
            this.time += 1;
            if (this.Player.isDeath) {
                this.scene = "start";
                this.Player.reset();
                this.time = 0;
                this.obstacles = [];
                this.high_score = Math.max(this.high_score, this.score);
                this.high_score_area.innerHTML = this.high_score;
                this.score = 0;
                this.input_data = [];
            }
        }
    }
    keyHandler(key) {
        if (key.key === " " && this.scene === "game") {
            this.Player.jump();
        }
        if (this.scene === "start") {
            this.scene = "game";
        }
    }
}

class CanvasObject {
    constructor() {
        this.canvas = document.getElementById("canvas");
        this.ctx = this.canvas.getContext("2d");
        this.GameScene = new GameScene();
    }

    async render() {
        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0, 0, WIDTH, HEIGHT);
        await this.GameScene.update();
        this.GameScene.render(this.ctx, this.canvas);
        requestAnimationFrame(this.render.bind(this));
    }
}

window.onload = () => {
    const canvasObject = new CanvasObject();
    window.requestAnimationFrame(canvasObject.render.bind(canvasObject));
};
