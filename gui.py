import tkinter as tk
from tkinter import messagebox
import cv2
from face_utils import enroll_user, recognize, capture_image_from_webcam

DEFAULT_LEVEL = 1  # nível padrão para simplificar

def cmd_enroll_gui():
    img = capture_image_from_webcam()
    name = simple_prompt("Digite o nome do usuário:")
    if name:
        user = enroll_user(name, DEFAULT_LEVEL, img)
        messagebox.showinfo(
            "Cadastro",
            f"Usuário cadastrado com sucesso!\n\nID: {user.id}\nNome: {user.name}\nNível: {user.role_level}"
        )

def cmd_login_gui():
    img = capture_image_from_webcam()
    user, dist, err = recognize(img)
    if user:
        messagebox.showinfo(
            "Autenticação",
            f"Autenticado com sucesso!\n\nNome: {user.name}\nID: {user.id}\nDistância: {dist:.4f}\nNível: {user.role_level}"
        )
    else:
        msg = err if err else f"Nenhuma correspondência confiável.\nMenor distância: {dist:.4f}"
        messagebox.showwarning("Autenticação", msg)

def simple_prompt(prompt):
    """Simples popup para input"""
    result = None

    def on_submit():
        nonlocal result
        val = entry.get().strip()
        if val:
            result = val
            popup.destroy()
        else:
            messagebox.showerror("Erro", "Digite um valor válido!")

    popup = tk.Toplevel()
    popup.title(prompt)
    popup.geometry("300x120")
    popup.resizable(False, False)

    tk.Label(popup, text=prompt, font=("Arial", 12)).pack(pady=10)
    entry = tk.Entry(popup, font=("Arial", 12))
    entry.pack(pady=5)
    tk.Button(popup, text="OK", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=on_submit).pack(pady=5)

    popup.grab_set()
    popup.wait_window()
    return result

# Janela principal
root = tk.Tk()
root.title("Sistema Biométrico")
root.geometry("350x200")
root.resizable(False, False)
root.configure(bg="#f0f0f0")

tk.Label(root, text="Bem-vindo ao Sistema Biométrico", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=15)

btn_style = {"font": ("Arial", 12, "bold"), "width": 20, "bd": 0}

tk.Button(root, text="CADASTRAR", bg="#2196F3", fg="white", **btn_style, command=cmd_enroll_gui).pack(pady=10)
tk.Button(root, text="ENTRAR", bg="#FF5722", fg="white", **btn_style, command=cmd_login_gui).pack(pady=10)

root.mainloop()
