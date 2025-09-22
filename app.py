import os
import torch
from torch import nn
from flask import Flask, render_template, request, session
from PIL import Image
from dcgan_arch import Generator, Discriminator
import torchvision.utils as vutils
import time
import secrets
import glob

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = secrets.token_hex(16)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate generator
gen = Generator(z_dim=100, ngf=128, nc=3).to(device)

def load_model():
    try:
        checkpoint = torch.load("models/best_gan4.pth", map_location=device)
        gen.load_state_dict(checkpoint["gen_state_dict"])
        gen.eval()
        print("Model loaded successfully")
        return gen
    except FileNotFoundError:
        print("Error: Model file 'models/best_gan4.pth' not found.")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

def clean_static_folder(max_files=200):  # Increased for testing
    files = sorted(glob.glob("static/generated_*.png"), key=os.path.getmtime)
    if len(files) > max_files:
        for old_file in files[:-max_files]:
            try:
                os.remove(old_file)
                print(f"Deleted old file: {old_file}")
            except Exception as e:
                print(f"Error deleting {old_file}: {e}")
    batch_files = sorted(glob.glob("static/batch/batch_*.png"), key=os.path.getmtime)
    if len(batch_files) > max_files:
        for old_file in batch_files[:-max_files]:
            try:
                os.remove(old_file)
                print(f"Deleted old batch file: {old_file}")
            except Exception as e:
                print(f"Error deleting {old_file}: {e}")

def save_generated_image(z, name="generated"):
    with torch.no_grad():
        fake_img = gen(z).cpu()
        print(f"Generated image shape: {fake_img.shape}, min: {fake_img.min()}, max: {fake_img.max()}")

    os.makedirs("static", exist_ok=True)

    # Overwrite based on provided name
    filename = f"{name}.png"
    save_path = os.path.join("static", filename)

    try:
        vutils.save_image(fake_img, save_path, normalize=True)
        print(f"Saved image to {save_path}")

        if os.path.exists(save_path):
            print(f"Confirmed: {save_path} exists")
        else:
            print(f"Error: {save_path} not found after saving")

    except Exception as e:
        print(f"Error saving image {save_path}: {e}")

    # Use timestamp only for cache-busting
    ts = int(time.time() * 1000)
    return filename, ts



def save_generated_batch(z_batch):
    # Make sure batch folder exists
    batch_folder = "static/batch"
    os.makedirs(batch_folder, exist_ok=True)

    # 1. Delete all old images in batch folder
    for old_file in glob.glob(os.path.join(batch_folder, "batch_*.png")):
        try:
            os.remove(old_file)
            print(f"Deleted old batch file: {old_file}")
        except Exception as e:
            print(f"Error deleting {old_file}: {e}")

    with torch.no_grad():
        fake_imgs = gen(z_batch).cpu()
        print(f"Generated batch shape: {fake_imgs.shape}, min: {fake_imgs.min()}, max: {fake_imgs.max()}")

    filenames = []
    for i in range(fake_imgs.size(0)):
        # 2. Save with fixed names 0–99
        filename = f"batch/batch_{i}.png"
        save_path = os.path.join("static", filename)
        try:
            vutils.save_image(fake_imgs[i], save_path, normalize=True)
            print(f"Saved batch image to {save_path}")
            filenames.append((filename, int(time.time() * 1000)))  # add ts for cache-busting
        except Exception as e:
            print(f"Error saving batch image {save_path}: {e}")

    return filenames


@app.after_request
def add_no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
    response.headers['Expires'] = '0'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    image1, image2, image_interp, image_urls = None, None, None, []
    # image1_ts = image2_ts = image_interp_ts = int(time.time() * 1000)

    image1_ts = session.get("image1_ts", int(time.time() * 1000))
    image2_ts = session.get("image2_ts", int(time.time() * 1000))
    image_interp_ts = session.get("image_interp_ts", int(time.time() * 1000))

    alpha = 0.5
    latent_dim = 100
    batch_size = 1
    action = None

    if "z1" not in session or "z2" not in session:
        session["z1"] = torch.randn(batch_size, latent_dim).tolist()
        session["z2"] = torch.randn(batch_size, latent_dim).tolist()

    if request.method == "POST":
        action = request.form.get("action")
        print(f"Action received: {action}")

        if action == "generate1":
            session["z1"] = torch.randn(batch_size, latent_dim).tolist()
            z1 = torch.tensor(session["z1"], device=device).view(batch_size, latent_dim, 1, 1)
            image1, image1_ts = save_generated_image(z1, name="generated1")
            session["image1"], session["image1_ts"] = image1, image1_ts

        elif action == "generate2":
            session["z2"] = torch.randn(batch_size, latent_dim).tolist()
            z2 = torch.tensor(session["z2"], device=device).view(batch_size, latent_dim, 1, 1)
            image2, image2_ts = save_generated_image(z2, name="generated2")
            session["image2"], session["image2_ts"] = image2, image2_ts

        elif action == "morph":
            alpha = float(request.form.get("alpha", 0.5))
            alpha = max(0.0, min(1.0, alpha))
            z1 = torch.tensor(session["z1"], device=device).view(batch_size, latent_dim, 1, 1)
            z2 = torch.tensor(session["z2"], device=device).view(batch_size, latent_dim, 1, 1)
            image1, image1_ts = save_generated_image(z1, name="generated1")
            image2, image2_ts = save_generated_image(z2, name="generated2")
            session["image1"], session["image1_ts"] = image1, image1_ts
            session["image2"], session["image2_ts"] = image2, image2_ts
            z_interp = (1 - alpha) * z1 + alpha * z2
            image_interp, image_interp_ts = save_generated_image(z_interp, name="generated_interp")
            session["image_interp"], session["image_interp_ts"] = image_interp, image_interp_ts


        elif action == "generate100":
            batch_size = 100
            z_batch = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            image_urls = save_generated_batch(z_batch)

            image1 = None
            image2 = None
            image_interp = None

        print(f"Rendering with: image1={image1}, image2={image2}, image_interp={image_interp}, image_urls={len(image_urls)} images")

    # Only reload session values if not batch mode
    if action != "generate100":
        # 🔑 always load from session, so they don’t vanish
        image1 = session.get("image1")
        image1_ts = session.get("image1_ts")
        image2 = session.get("image2")
        image2_ts = session.get("image2_ts")
        image_interp = session.get("image_interp")
        image_interp_ts = session.get("image_interp_ts")

    return render_template(
        "index.html",
        image1=image1,
        image2=image2,
        image_interp=image_interp,
        image1_ts=image1_ts,
        image2_ts=image2_ts,
        image_interp_ts=image_interp_ts,
        alpha=alpha,
        image_urls=image_urls
    )


from flask import jsonify

@app.route("/morph_ajax", methods=["POST"])
def morph_ajax():
    latent_dim = 100
    batch_size = 1
    if "z1" not in session or "z2" not in session:
        return jsonify({"error": "latent vectors not initialized"}), 400

    data = request.get_json() or {}
    alpha = max(0.0, min(1.0, float(data.get("alpha", 0.5))))

    z1 = torch.tensor(session["z1"], device=device).view(batch_size, latent_dim, 1, 1)
    z2 = torch.tensor(session["z2"], device=device).view(batch_size, latent_dim, 1, 1)

    z_interp = (1 - alpha) * z1 + alpha * z2
    filename, ts = save_generated_image(z_interp, name="generated_interp")

    session["image_interp"] = filename
    session["image_interp_ts"] = ts

    return jsonify({"filename": filename, "ts": ts, "alpha": round(alpha, 3)})



if __name__ == "__main__":
    app.run(debug=True)