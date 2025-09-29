import argparse
from face_utils import enroll_user, recognize, capture_image_from_webcam
from db import init_db

def cmd_enroll(args):
    print("Capture uma foto para enrolamento na webcam...")
    img = capture_image_from_webcam(show=True)
    user = enroll_user(args.name, args.level, img=img)
    print(f"Usuário cadastrado: id={user.id}, nome={user.name}, nível={user.role_level}")

def cmd_login(args):
    print("Realizando autenticação via webcam...")
    user, dist, err = recognize()
    if user:
        print(f"Autenticado: {user.name} (id={user.id}) — distância={dist:.4f} — nível={user.role_level}")
    else:
        if err:
            print("Falha:", err)
        else:
            print(f"Nenhuma correspondência confiável (menor distância encontrada: {dist})")

# reconhece é idêntico ao login
def cmd_recognize(args):
    cmd_login(args)

def main():
    init_db()
    parser = argparse.ArgumentParser(description="Sistema biométrico - CLI")
    sub = parser.add_subparsers(dest="cmd")

    # enroll
    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--name", required=True)
    p_enroll.add_argument("--level", type=int, required=True, choices=[1,2,3])

    # login
    sub.add_parser("login")

    # recognize
    sub.add_parser("recognize")

    args = parser.parse_args()
    if args.cmd == "enroll":
        cmd_enroll(args)
    elif args.cmd == "login":
        cmd_login(args)
    elif args.cmd == "recognize":
        cmd_recognize(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
