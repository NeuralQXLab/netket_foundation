# from netket.operator import FermionOperator2ndJax
# from netket.experimental.operator import (
#     ParticleNumberConservingFermioperator2nd,
#     ParticleNumberAndSpinConservingFermioperator2nd,
# )
# from netket.utils.deprecation import deprecated_new_name

# from netket_pro._src.monkeypatch.util import add_method

# # TODO : added in july 2025, remove at some point


# @add_method(ParticleNumberConservingFermioperator2nd)
# @classmethod
# @deprecated_new_name("from_fermionoperator2nd")
# def from_fermiop(cls, ha: FermionOperator2ndJax, **kwargs):
#     return cls.from_fermionoperator2nd(ha, **kwargs)


# @add_method(ParticleNumberAndSpinConservingFermioperator2nd)
# @classmethod
# @deprecated_new_name("from_fermionoperator2nd")
# def from_fermiop(cls, ha: FermionOperator2ndJax, **kwargs):  # noqa: F811
#     return cls.from_fermionoperator2nd(ha, **kwargs)
