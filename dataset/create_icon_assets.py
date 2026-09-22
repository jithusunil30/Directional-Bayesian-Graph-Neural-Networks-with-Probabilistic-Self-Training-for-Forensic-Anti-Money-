import os
from PIL import Image, ImageDraw, ImageFont

def create_icon_assets():
    base_dir = r"c:\Users\USER\OneDrive\Desktop\Probalistic Graphical Lab"
    icon_dir = os.path.join(base_dir, "dataset", "eda_plots", "icons")
    os.makedirs(icon_dir, exist_ok=True)

    size = (128, 128)

    # 1. Licit Normal Person Icon (Green background, person silhouette)
    im_lic = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im_lic)
    # Circle BG
    draw.ellipse([4, 4, 124, 124], fill="#2ECC71", outline="#145A32", width=4)
    # Head
    draw.ellipse([48, 24, 80, 56], fill="#FFFFFF")
    # Body
    draw.chord([32, 60, 96, 114], start=180, end=360, fill="#FFFFFF")
    im_lic.save(os.path.join(icon_dir, "licit_person.png"))

    # 2. Illicit Thief Icon (Red background, robber/hood/mask silhouette)
    im_ill = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im_ill)
    # Circle BG
    draw.ellipse([4, 4, 124, 124], fill="#E74C3C", outline="#78281F", width=4)
    # Hood / Head
    draw.ellipse([42, 22, 86, 66], fill="#111111")
    # Mask over eyes
    draw.rectangle([44, 40, 84, 52], fill="#FFFFFF")
    draw.ellipse([50, 43, 58, 49], fill="#111111")
    draw.ellipse([70, 43, 78, 49], fill="#111111")
    # Shoulders / Body
    draw.chord([28, 64, 100, 116], start=180, end=360, fill="#111111")
    im_ill.save(os.path.join(icon_dir, "illicit_thief.png"))

    # 3. Unknown Question Mark Icon (Purple background, white ?)
    im_unk = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im_unk)
    # Circle BG
    draw.ellipse([4, 4, 124, 124], fill="#9B59B6", outline="#4A235A", width=4)
    
    # Try loading a bold font, or draw vector question mark
    try:
        font = ImageFont.truetype("arialbd.ttf", 72)
        draw.text((64, 60), "?", fill="#FFFFFF", font=font, anchor="mm")
    except:
        # Fallback vector ?
        draw.arc([44, 24, 84, 64], start=210, end=360, fill="#FFFFFF", width=10)
        draw.line([(84, 44), (64, 70), (64, 80)], fill="#FFFFFF", width=10)
        draw.ellipse([59, 90, 69, 100], fill="#FFFFFF")

    im_unk.save(os.path.join(icon_dir, "unknown_question.png"))
    print(f"Created all 3 node icon assets in: {icon_dir}")

if __name__ == "__main__":
    create_icon_assets()
