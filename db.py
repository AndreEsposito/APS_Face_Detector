from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    role_level = Column(Integer, nullable=False)  # 1,2,3
    embedding = Column(String, nullable=False)    # store embedding as comma-separated string

DATABASE_URL = "sqlite:///biometria.db"
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def add_user(name: str, role_level: int, embedding_list):
    session = SessionLocal()
    emb_str = ",".join(map(str, embedding_list))
    user = User(name=name, role_level=role_level, embedding=emb_str)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.close()
    return user

def get_all_users():
    session = SessionLocal()
    users = session.query(User).all()
    session.close()
    return users

def find_user_by_id(uid: int):
    session = SessionLocal()
    u = session.query(User).filter(User.id == uid).first()
    session.close()
    return u
