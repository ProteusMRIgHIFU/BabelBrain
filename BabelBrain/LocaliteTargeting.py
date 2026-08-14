"""
LocaliteTargeting.py

A small helper library for reading, editing and writing "Targeting" XML files
used to describe transducer trajectories for an intervention.

XML structure
-------------
<Targeting ...attributes...>
    <Trajectory name="...">
        <InstrumentPose>
            <Matrix4D data00=... ... data33=.../>   # 4x4 pose matrix
        </InstrumentPose>
        <TargetPosition>
            <ColVec3D data0=.. data1=.. data2=../>   # focus position (3,)
        </TargetPosition>
        <Steering>
            <ColVec3D data0=.. data1=.. data2=../>   # steering vector (3,)
        </Steering>
    </Trajectory>
    ... more <Trajectory> ...
</Targeting>

Everything numeric is exposed as NumPy arrays:
    * Trajectory.pose             -> (4, 4) float array
    * Trajectory.target_position  -> (3,)   float array
    * Trajectory.steering         -> (3,)   float array
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np


# Default attributes written on the <Targeting> root when none are supplied.
DEFAULT_TARGETING_ATTRS = {
    "coordinateSpace": "RAS",
    "coordinateSystem": "NIfTI:S:Scanner",
    "creator": "Localite 4.0.0",
    "formatVersion": "v1.0",
}

# Comments emitted before each block on export, matching the sample file.
_POSE_COMMENTS = [
    "Transducer pose in Localite convention:",
    "x is pointing from transducer origin towards the head",
    "y is pointing from transducer origin towards the handle",
    "z is pointing from transducer origin to right (when the handle points to the bottom)",
]
_TARGET_COMMENT = "Focus position in image coordinates"
_STEERING_COMMENT = (
    "Beam steering parameters from mechanical displacement between transducer "
    "pose and focus position in Localite convention."
)


class Trajectory:
    """A single trajectory: a pose matrix, a target position and a steering vector."""

    _BBconvMat = np.array([[0, 0, -1, 0],                                                            
                                [0, -1, 0, 0],                                                            
                                [-1, 0, 0, 0],                                                                      
                                [0, 0, 0, 1]]).astype(float)

    def __init__(self, name, pose=None, target_position=None, steering=None):
        self.name = name
        self.pose = (
            np.eye(4) if pose is None else np.asarray(pose, dtype=float).reshape(4, 4)
        )
        self.target_position = (
            np.zeros(3)
            if target_position is None
            else np.asarray(target_position, dtype=float).reshape(3)
        )
        self.steering = (
            np.zeros(3)
            if steering is None
            else np.asarray(steering, dtype=float).reshape(3)
        )

    @property
    def babelbrain_affine(self):
        newMat=self.pose@self._BBconvMat
        newMat[:3,3]= self.target_position
        return newMat

    def update_localite_pose(self,adjustR=0.0,adjustA=0.0,adjustS=0.0):
        RASMat=np.zeros((4,4))
        RASMat[0,3]=adjustR
        RASMat[1,3]=adjustA
        RASMat[2,3]=adjustS
        #we just convert the adjustments to the pose
        newMat=RASMat@self._BBconvMat
        #then, we just apply the adjustment to the original pose coordinates
        self.pose[:3,3]+=newMat[:3,3]
    # ------------------------------------------------------------------ #
    # XML (de)serialization for a single <Trajectory> element
    # ------------------------------------------------------------------ #
    @classmethod
    def from_element(cls, elem):
        """Build a Trajectory from a <Trajectory> ElementTree element."""
        name = elem.get("name")

        matrix_elem = elem.find("./InstrumentPose/Matrix4D")
        if matrix_elem is None:
            raise ValueError(f"Trajectory '{name}' is missing InstrumentPose/Matrix4D")
        pose = np.array(
            [[float(matrix_elem.get(f"data{r}{c}")) for c in range(4)] for r in range(4)]
        )

        target_elem = elem.find("./TargetPosition/ColVec3D")
        if target_elem is None:
            raise ValueError(f"Trajectory '{name}' is missing TargetPosition/ColVec3D")
        target_position = np.array(
            [float(target_elem.get(f"data{i}")) for i in range(3)]
        )

        steering_elem = elem.find("./Steering/ColVec3D")
        if steering_elem is None:
            raise ValueError(f"Trajectory '{name}' is missing Steering/ColVec3D")
        steering = np.array([float(steering_elem.get(f"data{i}")) for i in range(3)])

        return cls(name, pose=pose, target_position=target_position, steering=steering)

    def to_element(self):
        """Serialize this trajectory to a <Trajectory> ElementTree element."""
        traj = ET.Element("Trajectory", {"name": self.name})

        for text in _POSE_COMMENTS:
            traj.append(ET.Comment(text))
        pose_elem = ET.SubElement(traj, "InstrumentPose")
        matrix_attrs = {
            f"data{r}{c}": repr(float(self.pose[r, c]))
            for r in range(4)
            for c in range(4)
        }
        ET.SubElement(pose_elem, "Matrix4D", matrix_attrs)

        traj.append(ET.Comment(_TARGET_COMMENT))
        target_elem = ET.SubElement(traj, "TargetPosition")
        ET.SubElement(
            target_elem,
            "ColVec3D",
            {f"data{i}": repr(float(self.target_position[i])) for i in range(3)},
        )

        traj.append(ET.Comment(_STEERING_COMMENT))
        steering_elem = ET.SubElement(traj, "Steering")
        ET.SubElement(
            steering_elem,
            "ColVec3D",
            {f"data{i}": repr(float(self.steering[i])) for i in range(3)},
        )

        return traj

    def __repr__(self):
        return (
            f"Trajectory(name={self.name!r}, "
            f"target_position={self.target_position.tolist()})"
        )
 
    def __str__(self):
        pose_str = np.array2string(self.pose, precision=6, suppress_small=True)
        # indent the matrix so every row lines up under "pose ="
        pose_str = pose_str.replace("\n", "\n         ")
        return (
            f"Trajectory(name={self.name!r},\n"
            f"  target_position={self.target_position.tolist()},\n"
            f"  steering={self.steering.tolist()},\n"
            f"  pose =\n         {pose_str})"
        )


class LocaliteTargeting:
    """A collection of trajectories plus the <Targeting> root attributes."""

    def __init__(self, trajectories=None, attributes=None):
        self.trajectories = list(trajectories) if trajectories else []
        self.attributes = dict(attributes) if attributes else dict(DEFAULT_TARGETING_ATTRS)

    # ------------------------------------------------------------------ #
    # 1) Reading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_string(cls, xml_string):
        """Parse a Targeting object from an XML string."""
        root = ET.fromstring(xml_string)
        return cls._from_root(root)

    @classmethod
    def from_file(cls, path):
        """Parse a Targeting object from an XML file on disk."""
        tree = ET.parse(path)
        return cls._from_root(tree.getroot())

    @classmethod
    def _from_root(cls, root):
        if root.tag != "Targeting":
            raise ValueError(f"Root element must be <Targeting>, got <{root.tag}>")
        trajectories = [
            Trajectory.from_element(elem) for elem in root.findall("Trajectory")
        ]
        return cls(trajectories=trajectories, attributes=dict(root.attrib))

    # ------------------------------------------------------------------ #
    # 2) / 3) Accessing and modifying trajectories
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    @property
    def names(self):
        return [t.name for t in self.trajectories]

    def get(self, name):
        """Return the trajectory with the given name (or None if absent)."""
        for t in self.trajectories:
            if t.name == name:
                return t
        return None

    def __getitem__(self, key):
        """Index by integer position or by trajectory name."""
        if isinstance(key, int):
            return self.trajectories[key]
        traj = self.get(key)
        if traj is None:
            raise KeyError(key)
        return traj

    def modify(self, name, *, new_name=None, pose=None, target_position=None, steering=None):
        """Modify fields of an existing trajectory identified by ``name``."""
        traj = self.get(name)
        if traj is None:
            raise KeyError(f"No trajectory named {name!r}")
        if new_name is not None:
            traj.name = new_name
        if pose is not None:
            traj.pose = np.asarray(pose, dtype=float).reshape(4, 4)
        if target_position is not None:
            traj.target_position = np.asarray(target_position, dtype=float).reshape(3)
        if steering is not None:
            traj.steering = np.asarray(steering, dtype=float).reshape(3)
        return traj

    # ------------------------------------------------------------------ #
    # 4) Adding / removing trajectories
    # ------------------------------------------------------------------ #
    def add(self, trajectory=None, *, name=None, pose=None,
            target_position=None, steering=None, replace=False):
        """Add a trajectory, either a ready-made Trajectory or from its fields."""
        if trajectory is None:
            if name is None:
                raise ValueError("Provide either a Trajectory or at least a name.")
            trajectory = Trajectory(
                name, pose=pose, target_position=target_position, steering=steering
            )
        if self.get(trajectory.name) is not None:
            if not replace:
                raise ValueError(
                    f"A trajectory named {trajectory.name!r} already exists "
                    f"(pass replace=True to overwrite)."
                )
            self.remove(trajectory.name)
        self.trajectories.append(trajectory)
        return trajectory

    def remove(self, name):
        """Remove the trajectory with the given name. Returns it."""
        traj = self.get(name)
        if traj is None:
            raise KeyError(f"No trajectory named {name!r}")
        self.trajectories.remove(traj)
        return traj

    # ------------------------------------------------------------------ #
    # 5) Writing
    # ------------------------------------------------------------------ #
    def to_element(self):
        root = ET.Element("Targeting", self.attributes)
        for traj in self.trajectories:
            root.append(traj.to_element())
        return root

    def to_string(self, pretty=True, encoding="UTF-8"):
        """Serialize to an XML string mirroring the input structure."""
        root = self.to_element()
        rough = ET.tostring(root, encoding="unicode")
        if not pretty:
            header = (
                f'<?xml version="1.0" encoding="{encoding}" standalone="no"?>\n'
            )
            return header + rough
        parsed = minidom.parseString(rough)
        pretty_xml = parsed.toprettyxml(indent="  ", encoding=encoding)
        return pretty_xml.decode(encoding)

    def to_file(self, path, pretty=True, encoding="UTF-8"):
        """Write the trajectories to an XML file."""
        text = self.to_string(pretty=pretty, encoding=encoding)
        with open(path, "w", encoding=encoding) as fh:
            fh.write(text)

    def __repr__(self):
        return f"Targeting({len(self)} trajectories: {self.names})"

    # ------------------------------------------------------------------ #
    # 6) Extra tools to work with BabelBrain
    # ------------------------------------------------------------------ #
    def ReturnBabelBrainTrajectories(self,bGetID=False):
        '''
        This will build similar as ReadTrajectoryBrainsight.
        It will return either a 4x4 BabelBrain compatible matrix if only one trajectory, 
        or a 4x4xn trajectories. Along with the ID(s, trajectory names) if requested
        '''
        if len(self.trajectories)==0:
            raise RuntimeError("trajectories must have at least one entry")
        if len(self.trajectories)==1:
            ID=self.trajectories[0].name
            mat = self.trajectories[0].babelbrain_affine
        else:
            mat=np.zeros((4,4,len(self.trajectories)))
            ID=[]
            for n in range(len(self.trajectories)):
                mat[:,:,n]=self.trajectories[n].babelbrain_affine
                ID.append(self.trajectories[n].name)
        if bGetID:
            return mat, ID
        else:
            return mat

if __name__ == "__main__":
    # Minimal round-trip demonstration.
    sample = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Targeting coordinateSpace="RAS" coordinateSystem="NIfTI:S:Scanner" creator="Localite 4.0.0" formatVersion="v1.0">
  <Trajectory name="RightVIM_12_Aug_2026_182005">
    <InstrumentPose>
      <Matrix4D data00="-0.6959298521446974" data01="0.7007707611210828" data02="0.15685045639397263" data03="63.69349285633857" data10="-0.048330710376095765" data11="0.17221900910032562" data12="-0.9838723696488301" data13="17.971949975349332" data20="-0.7164815892594598" data21="-0.6922868174980943" data22="-0.08598359721457643" data23="80.8490956348583" data30="0.0" data31="0.0" data32="0.0" data33="1.0"/>
    </InstrumentPose>
    <TargetPosition>
      <ColVec3D data0="15.3" data1="7.5" data2="17.6"/>
    </TargetPosition>
    <Steering>
      <ColVec3D data0="79.50139895783838" data1="8.070300775052136" data2="8.15090487123551"/>
    </Steering>
  </Trajectory>
</Targeting>"""

    tg = LocaliteTargeting.from_string(sample)
    print(tg)
    print("Pose:\n", tg["RightVIM_12_Aug_2026_182005"].pose)
    print("element 0\n",tg[0])
    print("Potarget_position:\n", tg["RightVIM_12_Aug_2026_182005"].target_position)
    BBMat=tg["RightVIM_12_Aug_2026_182005"].babelbrain_affine
    print('BabelBrain affine:\n',BBMat)
    tg["RightVIM_12_Aug_2026_182005"].update_localite_pose(BBMat)
    print('updated pose from BBMat (should be the same as before)\n',tg["RightVIM_12_Aug_2026_182005"].pose)

    # # modify, add, remove
    # tg.modify("RightVIM_12_Aug_2026_182005", target_position=[1, 2, 3])
    # tg.add(name="LeftVIM_demo", target_position=[4, 5, 6], steering=[0.1, 0.2, 0.3])
    # print(tg)
    # tg.remove("LeftVIM_demo")
    print(tg)
    print(tg.to_string())