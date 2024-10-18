from ase.io import read, write, Trajectory
from ase import Atoms
from ase.mep.neb import NEB as aseNEB
from ase.mep.neb import DyNEB
from ase.optimize import FIRE, BFGS
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
from copy import deepcopy

class NEB:
    def __init__(self, 
                 mlp_calc,
                 dft_calc,
                 init_file: str,
                 final_file: str,
                 nimages: int=20,
                 prefix: str='neb',
                 fmt: str=None,
                 autosort_tol: float=None,
                 spring_constant: float=0.1,
                 neb_algo: str="improvedtangent",
                 climb: bool=True,
                 dyneb: bool=False,
                 optimizer: str='BFGS',
                 fmax: float=0.05) -> None:
        '''
        
        Parameters:
        -----------
        mlp_calc: object
            The trained MLP calculator object.
        dft_calc: object
            The DFT calculator object.
        init_file: str
            The initial structure file.
        final_file: str
            The final structure file.
        nimages: int
            The number of images.
        prefix: str
            The prefix of the output files.
        fmt: str
            The format of the input files.
        autosort_tol: float
            The tolerance for autosort.
        spring_constant: float
            The spring constant.
        neb_algo: str
            The neb algorithm.
        climb: bool
            Whether to use the climbing image.
        dyneb: bool
            Whether to use the DyNEB.
        optimizer: str
            The optimizer to use. It can be 'BFGS' or 'FIRE'.
        fmax: float
            The maximum force.
        '''
        
        # setting about calculator
        self.mlp_calc = mlp_calc
        self.dft_calc = dft_calc

        # setting about interpolation
        self.init_file = init_file
        self.final_file = final_file
        self.fmt = fmt
        self.nimages = nimages
        self.prefix = prefix
        self.autosort_tol = autosort_tol

        # setting about neb calculation
        self.spring_constant = spring_constant
        self.neb_algo = neb_algo
        self.climb = climb
        self.dyneb = dyneb
        self.fmax = fmax
        if optimizer == 'BFGS':
            self.optimizer = BFGS
        elif optimizer == 'FIRE':
            self.optimizer = FIRE
        else:
            raise ValueError(f"Optimizer {optimizer} is not supported.")
        
        # check prefix folder
        self.p = Path(self.prefix)
        if not self.p.exists():
            self.p.mkdir(exist_ok=True)
        
    def __interpolate_path(self) -> list[Atoms]:
        is_atom = read(self.init_file, format=self.fmt)
        fs_atom = read(self.final_file, format=self.fmt)
        is_pmg = AseAtomsAdaptor.get_structure(is_atom)
        fs_pmg = AseAtomsAdaptor.get_structure(fs_atom)

        # Interpolate the path
        path = is_pmg.interpolate(fs_pmg, self.nimages + 1, autosort_tol=self.autosort_tol)
        path = [AseAtomsAdaptor.get_atoms(s) for s in path]

        return path

    def __set_neb(self, path: list[Atoms]) -> aseNEB | DyNEB:
        # set up neb calculation
        if self.dyneb == True:
            neb = DyNEB(
                images=path,
                climb=self.climb,
                dynamic_relaxation=True,
                fmax=self.fmax,
                method=self.neb_algo,
                parallel=False,
                scale_fmax=1.0,
                k=self.spring_constant,
            )
        else:
            neb = aseNEB(
                images=path,
                climb=self.climb,
                method=self.neb_algo,
                k=self.spring_constant,
                parallel=False,
            )
        return neb
    
    def __run_mlp(self, path: list[Atoms]) -> None:

        # 1. attach the mlp calculator
        mlp_path = deepcopy(path)
        for i in mlp_path[1:-1]:
            i.calc = self.mlp_calc
        
        # 2. set up neb calculation
        neb = self.__set_neb(mlp_path)
        
        # 3. run calculation
        traj = Trajectory(self.p / 'mlp.traj', 'w', neb)
        opt = self.optimizer(neb, trajectory=traj)
        opt.run(fmax=self.fmax)
            

    def __run_dft(self):
        pass 

    def run(self):
        '''
        1. Interpolate the path, using pymatgen.
        2. set up neb calculation.
        3. run neb calculation, use mlp calculator.
        4. for optimizaed images, run dft scf calculation, using dft calculator.
        5. post process, such as plot the energy profile.
        6. output all necessary files into the self.prefix folder.
        '''
        path = self.__interpolate_path()
        self.__run_mlp(path)
        pass