"""Modify Prefernces Columns

Revision ID: 16ef585bd9fc
Revises: 1291c707c325
Create Date: 2024-06-30 22:55:09.822057

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '16ef585bd9fc'
down_revision = '1291c707c325'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('preferences', schema=None) as batch_op:
        batch_op.alter_column('mem_size',
               existing_type=sa.INTEGER(),
               type_=sa.String(length=50),
               existing_nullable=True)
        batch_op.alter_column('release_year',
               existing_type=sa.INTEGER(),
               type_=sa.String(length=50),
               existing_nullable=True)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('preferences', schema=None) as batch_op:
        batch_op.alter_column('release_year',
               existing_type=sa.String(length=50),
               type_=sa.INTEGER(),
               existing_nullable=True)
        batch_op.alter_column('mem_size',
               existing_type=sa.String(length=50),
               type_=sa.INTEGER(),
               existing_nullable=True)

    # ### end Alembic commands ###
