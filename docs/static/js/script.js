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
        if (this.pos.x + this.size.x < 0) {
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
        if (this.pos.x + this.size.x < 0) {
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
        this.pos.x += this.velocity.x;
        this.pos.y += this.velocity.y;
        this.velocity.y += this.#G;
        this.velocity.y = Math.max(this.velocity.y, -8);
        this.velocity.y = Math.min(this.velocity.y, 10);

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
    static random = new Random();
    static player_type = 0;
    constructor(playerBtn) {
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
    }
    async render(ctx, canvas) {
        this.obstacles.forEach((e) => {
            e.draw(ctx, "obstacle");
        });
        this.Player.draw(ctx);
        ctx.font = "30px Brush Script MT";
        ctx.fillStyle = "white";
        const text = "SCORE: " + this.score;
        ctx.fillText(text, 200 - ctx.measureText(text).width / 2, 30);
    }
    async update(act = 0) {
        if (this.scene === "game") {
            if (act) this.Player.jump();
            let state = [0];
            this.obstacles.forEach((e) => {
                e.update();
                if (e.tag === "UP") {
                    state.push(e.size.y / 400);
                    state.push((e.size.y + e.size.x) / 400);
                    state.push((e.size.y + e.size.x + 120) / 400);
                    state.push((e.size.y + 120) / 400);
                }
                if (e.collides(this.Player)) {
                    if (e.tag === "OK") {
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
            this.Player.update();

            if (this.time % 50 == 0) {
                const rand = GameScene.random.nextInt(0, 400 - 150);
                this.obstacles.push(new Obstacle(WIDTH, 0, 40, rand, "UP"));
                this.obstacles.push(
                    new Obstacle(WIDTH, rand + 120, 40, HEIGHT, "DOWN")
                );
                this.obstacles.push(
                    new Point(WIDTH + 25 + 15, rand, 40, 120, "OK")
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
                GameScene.random = new Random(
                    Math.floor(Math.random() * 1000000 + 1)
                );
            }
            for (let i = state.length; i < 13; i++) {
                state.push(0);
            }
            state[0] = this.Player.pos.y / 400;
            this.input_data = stack_frame(state, this.input_data);
        }
    }
    keyHandler(key) {
        if (
            key.key === " " &&
            this.scene === "game" &&
            GameScene.player_type === 0
        ) {
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
        const playerBtn = document.getElementById("play");
        this.GameScene = new GameScene(playerBtn);
        this.GameScene.input_data = Array(65).fill(0);
        this.GameScene.input_data[65 - 13] = this.GameScene.Player.pos.y / 400;
        this.model = new PPO();
        playerBtn.addEventListener("change", (e) => {
            GameScene.player_type = Number(e.target.value);
        });
    }

    async render() {
        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0, 0, WIDTH, HEIGHT);
        const data = Float32Array.from(this.GameScene.input_data);
        const data_tensor = new ort.Tensor("float32", data, [1, 65]);
        if (this.GameScene.scene === "game") {
            let act = 0;

            if (GameScene.player_type === 1) {
                if (this.model.loaded) {
                    act = await this.model.predict(data_tensor);
                    act = Math.random() > act[0];
                }
            }
            await this.GameScene.update(act);
        }
        this.GameScene.render(this.ctx, this.canvas);
        requestAnimationFrame(this.render.bind(this));
    }
}

window.onload = () => {
    const canvasObject = new CanvasObject();
    window.requestAnimationFrame(canvasObject.render.bind(canvasObject));
};
