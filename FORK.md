# Why this fork exists

`mmr_perception` runs this filter to drop the UF850 arm and its tool out of the
Livox Mid-360 cloud before odometry and SLAM see it
(`mmr_perception/config/filter/arm_self_filter.yaml`). Upstream could not do that
job on this robot as released. The patches below are what it needed.

**Base:** upstream's `dev/mid360_point_type` at `8624b06`: `v1.1.0`, plus upstream's
per-link `shadow` switch (on `main`) and its Livox Mid-360 point type (not merged
yet). Our changes live on `sakar/v1.1.0`, on top of that base. Remotes: `origin`
is this fork (sakar-robotics), `upstream` is leggedrobotics.

## Patches

1. **Mesh links get scale and padding** (`bodies.h/.cpp`, `self_mask.h`).
   Upstream left `ConvexMesh` scale/padding as placeholders. Every UF850 link and
   the tool is a mesh, and the Mid-360's 2-3 cm range noise puts up to half of
   the returns off an unpadded hull's surface outside it. Padding now moves each
   hull face out along its normal, so a long link is padded as much along its
   sides as at its ends. The ray test now clips the ray against the faces'
   half-spaces, which is exact for the padded hull and replaces the per-triangle
   loop.
2. **The URDF's `<mesh scale>` is applied** (`self_mask.h`). Upstream ignored it.
   Our description has a mesh drawn in millimetres (`tilt_mount_plate.stl`,
   `scale="0.001 ..."`), whose hull would otherwise be 1000x too big and swallow
   the whole cloud.
3. **A link is skipped until its first transform arrives** (`self_mask.h`).
   Before that, a body sits at the cloud frame's origin, which is the lidar
   itself, and in shadow mode every ray ends inside it. That emptied every cloud
   until TF arrived, and for good if one link's TF never came (a base-only robot).
   A failed lookup now also warns (throttled), and only the first one per cloud
   waits the 100 ms timeout, so a stalled `joint_states` stalls each cloud once,
   not once per link.
4. **Output QoS can be overridden** (`self_filter.cpp`). The cloud publisher
   honours `qos_overrides.<topic>.publisher.*`. rtabmap subscribes RELIABLE, and
   upstream's fixed BEST_EFFORT output matched nothing. The default is unchanged.
5. **README**: resolved the merge-conflict markers committed with upstream's
   Mid-360 change, and documented the above.

Every patch has a regression test in `test/test_bodies.cpp` or
`test/test_self_filter.cpp`. The Mid-360 layout has a test too, against the exact
packed PointCloud2 that `livox_ros_driver2` publishes.

## Getting back to upstream

Each patch is self-contained, and each would make a reasonable upstream PR. Once
upstream has the ones `arm_self_filter.yaml` depends on, rebase onto that release
or drop the fork. Patches 1, 3 and 4 are the ones it depends on. Check with:

```bash
git log --oneline upstream/dev/mid360_point_type..sakar/v1.1.0
```
