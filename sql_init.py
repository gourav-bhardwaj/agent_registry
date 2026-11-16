from sqlalchemy import ForeignKey, create_engine, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

class Base(DeclarativeBase):
    pass

# Define the table structure using a Python class
class Department(Base):
    __tablename__ = 'department'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False)

class Address(Base):
    __tablename__ = 'address'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    address_1: Mapped[str] = mapped_column(String(200), nullable=True)
    address_2: Mapped[str] = mapped_column(String(200), nullable=True)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)

class Employee(Base):
    __tablename__ = 'employee'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    personal_email: Mapped[str] = mapped_column(String(100), nullable=False)
    official_email: Mapped[str] = mapped_column(String(100), nullable=False)
    address_id: Mapped[int] = mapped_column(
        ForeignKey(column="address.id", onupdate="CASCADE"),
        nullable=False
    )
    department_id: Mapped[int] = mapped_column(
        ForeignKey(column="department.id", ondelete="SET NULL"),
        nullable=True
    )

# Database Setup (In-Memory SQLite for testing)
engine = create_engine("sqlite:///:memory:", echo=True)
Base.metadata.create_all(engine)