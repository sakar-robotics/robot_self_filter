// ============================ bodies.cpp (Fixed) ============================
#include "robot_self_filter/bodies.h"
#include <LinearMath/btConvexHull.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace robot_self_filter
{
namespace bodies
{

namespace detail
{
  struct intersc
  {
    intersc(const tf2::Vector3 &_pt, double _tm) : pt(_pt), time(_tm) {}
    tf2::Vector3 pt;
    double time;
  };

  struct interscOrder
  {
    bool operator()(const intersc &a, const intersc &b) const
    {
      return a.time < b.time;
    }
  };
}

Body *createBodyFromShape(const shapes::Shape *shape)
{
  if (!shape) return nullptr;
  Body *body = nullptr;
  switch (shape->type)
  {
    case shapes::BOX:
      body = new Box(shape);
      break;
    case shapes::SPHERE:
      body = new Sphere(shape);
      break;
    case shapes::CYLINDER:
      body = new Cylinder(shape);
      break;
    case shapes::MESH:
      body = new ConvexMesh(shape);
      break;
    default:
      std::cerr << "[createBodyFromShape] Unknown shape type: " << shape->type << std::endl;
      break;
  }
  return body;
}

void mergeBoundingSpheres(const std::vector<BoundingSphere> &spheres, BoundingSphere &mergedSphere)
{
  if (spheres.empty())
  {
    mergedSphere.center.setValue(0.0, 0.0, 0.0);
    mergedSphere.radius = 0.0;
    return;
  }
  mergedSphere = spheres[0];
  for (size_t i = 1; i < spheres.size(); ++i)
  {
    if (spheres[i].radius <= 0.0) continue;
    double d = spheres[i].center.distance(mergedSphere.center);

    if (d + mergedSphere.radius <= spheres[i].radius)
    {
      mergedSphere.center = spheres[i].center;
      mergedSphere.radius = spheres[i].radius;
    }
    else if (d + spheres[i].radius > mergedSphere.radius)
    {
      tf2::Vector3 delta = mergedSphere.center - spheres[i].center;
      double new_radius   = (delta.length() + spheres[i].radius + mergedSphere.radius) / 2.0;
      mergedSphere.center = delta.normalized() * (new_radius - spheres[i].radius) + spheres[i].center;
      mergedSphere.radius = new_radius;
    }
  }
}

// Sphere
static const double ZERO = 1e-9;
static inline double distanceSQR(const tf2::Vector3 &p,
                                 const tf2::Vector3 &origin,
                                 const tf2::Vector3 &dir)
{
  tf2::Vector3 a = p - origin;
  double d = dir.dot(a);
  return a.length2() - d * d;
}

bool Sphere::containsPoint(const tf2::Vector3 &p, bool) const
{
  return (m_center - p).length2() < m_radius2;
}
void Sphere::useDimensions(const shapes::Shape *shape)
{
  m_radius = static_cast<const shapes::Sphere *>(shape)->radius;
}
void Sphere::updateInternalData()
{
  m_radiusU = (m_radius * m_scale) + m_padding;
  m_radius2 = m_radiusU * m_radiusU;
  m_center  = m_pose.getOrigin();
}
double Sphere::computeVolume() const
{
  return (4.0 * M_PI * m_radiusU * m_radiusU * m_radiusU) / 3.0;
}
void Sphere::computeBoundingSphere(BoundingSphere &sphere) const
{
  sphere.center = m_center;
  sphere.radius = m_radiusU;
}
bool Sphere::intersectsRay(const tf2::Vector3 &origin,
                           const tf2::Vector3 &dir,
                           std::vector<tf2::Vector3> *intersections,
                           unsigned int count) const
{
  if (distanceSQR(m_center, origin, dir) > m_radius2) return false;

  bool result = false;
  tf2::Vector3 cp = origin - m_center;
  double dpcpv     = cp.dot(dir);
  tf2::Vector3 w   = cp - dpcpv * dir;
  tf2::Vector3 Q   = m_center + w;
  double x         = m_radius2 - w.length2();
  if (std::fabs(x) < ZERO)
  {
    w = Q - origin;
    double dpQv = w.dot(dir);
    if (dpQv > ZERO)
    {
      if (intersections) intersections->push_back(Q);
      result = true;
    }
  }
  else if (x > 0.0)
  {
    x = std::sqrt(x);
    w = dir * x;
    tf2::Vector3 A = Q - w;
    tf2::Vector3 B = Q + w;
    double dpAv = (A - origin).dot(dir);
    double dpBv = (B - origin).dot(dir);
    if (dpAv > ZERO)
    {
      result = true;
      if (intersections)
      {
        intersections->push_back(A);
        if (count == 1) return true;
      }
    }
    if (dpBv > ZERO)
    {
      result = true;
      if (intersections) intersections->push_back(B);
    }
  }
  return result;
}

// Cylinder
bool Cylinder::containsPoint(const tf2::Vector3 &p, bool) const
{
  tf2::Vector3 v = p - m_center;
  double pH = v.dot(m_normalH);
  if (std::fabs(pH) > m_length2) return false;
  double pB1 = v.dot(m_normalB1);
  double remain = m_radius2 - pB1 * pB1;
  if (remain < 0.0) return false;
  double pB2 = v.dot(m_normalB2);
  return (pB2 * pB2 < remain);
}
void Cylinder::useDimensions(const shapes::Shape *shape)
{
  m_length = static_cast<const shapes::Cylinder *>(shape)->length;
  m_radius = static_cast<const shapes::Cylinder *>(shape)->radius;
}
void Cylinder::updateInternalData()
{
  // Vertical direction scaled by m_scale_vert, radial by m_scale_rad
  m_radiusU = (m_radius * m_scale_rad) + m_padding_rad;
  m_radius2 = m_radiusU * m_radiusU;
  m_length2 = ((m_length * 0.5) * m_scale_vert) + m_padding_vert;
  m_center  = m_pose.getOrigin();

  m_radiusBSqr = m_length2 * m_length2 + m_radius2;
  m_radiusB    = std::sqrt(m_radiusBSqr);

  tf2::Matrix3x3 basis(m_pose.getBasis());
  m_normalB1 = basis.getColumn(0);
  m_normalB2 = basis.getColumn(1);
  m_normalH  = basis.getColumn(2);

  double tmp = -m_normalH.dot(m_center);
  m_d1 = tmp + m_length2;
  m_d2 = tmp - m_length2;
}
double Cylinder::computeVolume() const
{
  return M_PI * m_radius2 * (m_length2 * 2.0);
}
void Cylinder::computeBoundingSphere(BoundingSphere &sphere) const
{
  sphere.center = m_center;
  sphere.radius = m_radiusB;
}
bool Cylinder::intersectsRay(const tf2::Vector3 &origin,
                             const tf2::Vector3 &dir,
                             std::vector<tf2::Vector3> *intersections,
                             unsigned int count) const
{
  if (distanceSQR(m_center, origin, dir) > m_radiusBSqr) return false;

  // Caps
  std::vector<detail::intersc> ipts;
  double tmp = m_normalH.dot(dir);
  if (std::fabs(tmp) > ZERO)
  {
    double tmp2 = -m_normalH.dot(origin);
    double t1   = (tmp2 - m_d1) / tmp;
    if (t1 > 0.0)
    {
      tf2::Vector3 p1 = origin + dir * t1;
      tf2::Vector3 v1 = p1 - m_center; 
      v1 = v1 - m_normalH.dot(v1) * m_normalH;
      if (v1.length2() < (m_radius2 + ZERO))
      {
        if (!intersections) return true;
        ipts.emplace_back(p1, t1);
      }
    }
    double t2 = (tmp2 - m_d2) / tmp;
    if (t2 > 0.0)
    {
      tf2::Vector3 p2 = origin + dir * t2;
      tf2::Vector3 v2 = p2 - m_center;
      v2 = v2 - m_normalH.dot(v2) * m_normalH;
      if (v2.length2() < (m_radius2 + ZERO))
      {
        if (!intersections) return true;
        ipts.emplace_back(p2, t2);
      }
    }
  }

  // Barrel
  if (ipts.size() < 2)
  {
    tf2::Vector3 VD  = m_normalH.cross(dir);
    tf2::Vector3 ROD = m_normalH.cross(origin - m_center);
    double a = VD.length2();
    double b = 2.0 * ROD.dot(VD);
    double c = ROD.length2() - m_radius2;
    double d = b * b - 4.0 * a * c;
    if ((d > 0.0) && (std::fabs(a) > ZERO))
    {
      d = std::sqrt(d);
      double e = 2.0 * a;
      double t1 = (-b + d) / e;
      double t2 = (-b - d) / e;
      if (t1 > 0.0)
      {
        tf2::Vector3 p1 = origin + dir * t1;
        tf2::Vector3 v1 = m_center - p1;
        if (std::fabs(m_normalH.dot(v1)) < (m_length2 + ZERO))
        {
          if (!intersections) return true;
          ipts.emplace_back(p1, t1);
        }
      }
      if (t2 > 0.0)
      {
        tf2::Vector3 p2 = origin + dir * t2;
        tf2::Vector3 v2 = m_center - p2;
        if (std::fabs(m_normalH.dot(v2)) < (m_length2 + ZERO))
        {
          if (!intersections) return true;
          ipts.emplace_back(p2, t2);
        }
      }
    }
  }
  if (ipts.empty()) return false;
  std::sort(ipts.begin(), ipts.end(), detail::interscOrder());
  const size_t n = count > 0 ?
    std::min(static_cast<size_t>(count), ipts.size()) : ipts.size();
  for (size_t i = 0; i < n; ++i)
    intersections->push_back(ipts[i].pt);
  return true;
}

// Box
bool Box::containsPoint(const tf2::Vector3 &p, bool) const
{
  tf2::Vector3 v = p - m_center;
  double pL = v.dot(m_normalL); if (std::fabs(pL) > m_length2) return false;
  double pW = v.dot(m_normalW); if (std::fabs(pW) > m_width2)  return false;
  double pH = v.dot(m_normalH); if (std::fabs(pH) > m_height2) return false;
  return true;
}
void Box::useDimensions(const shapes::Shape *shape)
{
  const double *size = static_cast<const shapes::Box *>(shape)->size;
  m_length = size[0];
  m_width  = size[1];
  m_height = size[2];
}
void Box::updateInternalData()
{
  // scale + padding per dimension
  double halfL = 0.5 * m_length * m_scale[0] + m_padding[0];
  double halfW = 0.5 * m_width  * m_scale[1] + m_padding[1];
  double halfH = 0.5 * m_height * m_scale[2] + m_padding[2];

  m_length2 = halfL;
  m_width2  = halfW;
  m_height2 = halfH;

  m_center  = m_pose.getOrigin();
  m_radius2 = (halfL * halfL) + (halfW * halfW) + (halfH * halfH);
  m_radiusB = std::sqrt(m_radius2);

  tf2::Matrix3x3 basis(m_pose.getBasis());
  m_normalL = basis.getColumn(0);
  m_normalW = basis.getColumn(1);
  m_normalH = basis.getColumn(2);

  tf2::Vector3 tmp = m_normalL * halfL + m_normalW * halfW + m_normalH * halfH;
  m_corner1 = m_center - tmp;
  m_corner2 = m_center + tmp;
}
double Box::computeVolume() const
{
  return 8.0 * m_length2 * m_width2 * m_height2;
}
void Box::computeBoundingSphere(BoundingSphere &sphere) const
{
  sphere.center = m_center;
  sphere.radius = m_radiusB;
}
bool Box::intersectsRay(const tf2::Vector3 &origin,
                        const tf2::Vector3 &dir,
                        std::vector<tf2::Vector3> *intersections,
                        unsigned int count) const
{
  const tf2::Vector3 center_offset = m_center - origin;
  const double center_projection = dir.dot(center_offset);
  if (center_projection < -m_radiusB ||
      center_offset.length2() - center_projection * center_projection > m_radius2)
  {
    return false;
  }

  double t_near = -std::numeric_limits<double>::infinity();
  double t_far  = std::numeric_limits<double>::infinity();

  // Project the ray into the box's local basis. The previous implementation
  // recomputed both world-space slab planes from the two corners for every
  // ray (twelve dot products total). Local symmetric extents need only the
  // origin and direction projections (six dot products total).
  const tf2::Vector3 local_origin = origin - m_center;
  const auto update_slab = [&](const tf2::Vector3 &normal, const double half_extent)
  {
    const double direction_projection = normal.dot(dir);
    const double origin_projection = normal.dot(local_origin);
    if (std::fabs(direction_projection) > ZERO)
    {
      double t1 = (-half_extent - origin_projection) / direction_projection;
      double t2 = (half_extent - origin_projection) / direction_projection;
      if (t1 > t2) std::swap(t1, t2);
      if (t1 > t_near) t_near = t1;
      if (t2 < t_far)  t_far  = t2;
      return t_near <= t_far && t_far >= 0.0;
    }
    return origin_projection >= -half_extent && origin_projection <= half_extent;
  };

  if (!update_slab(m_normalL, m_length2) ||
      !update_slab(m_normalW, m_width2) ||
      !update_slab(m_normalH, m_height2))
    return false;
  if (intersections)
  {
    if ((t_far - t_near) > ZERO)
    {
      intersections->push_back(origin + dir * t_near);
      if (count > 1) intersections->push_back(origin + dir * t_far);
    }
    else
    {
      intersections->push_back(origin + dir * t_far);
    }
  }
  return true;
}

// ConvexMesh
ConvexMesh::ConvexMesh() : Body()
{
  m_type        = shapes::MESH;
  m_scale       = 1.0;
  m_padding     = 0.0;
  m_meshRadiusB = 0.0;
  m_radiusB     = 0.0;
  m_radiusBSqr  = 0.0;
}
ConvexMesh::ConvexMesh(const shapes::Shape *shape) : ConvexMesh()
{
  setDimensions(shape);
}
void ConvexMesh::setScale(double scale)
{
  m_scale = scale;
  updateScaledGeometry();
  updateInternalData();
}
void ConvexMesh::setPadding(double padding)
{
  m_padding = padding;
  updateScaledGeometry();
  updateInternalData();
}
bool ConvexMesh::containsPoint(const tf2::Vector3 &p, bool) const
{
  if (!m_boundingBox.containsPoint(p)) return false;
  return isPointInsidePlanes(m_iPose * p);
}
bool ConvexMesh::intersectsRay(const tf2::Vector3 &origin,
                               const tf2::Vector3 &dir,
                               std::vector<tf2::Vector3> *intersections,
                               unsigned int count) const
{
  if (m_scaledPlanes.empty()) return false;
  if (distanceSQR(m_center, origin, dir) > m_radiusBSqr) return false;
  if (!m_boundingBox.intersectsRay(origin, dir)) return false;

  // The hull is the intersection of its faces' half-spaces, so the ray is inside it from the
  // last face it enters through to the first face it leaves through.
  const tf2::Vector3 orig = m_iPose * origin;
  const tf2::Vector3 dr = tf2::Matrix3x3(m_iPose.getBasis()) * dir;
  double t_near = -std::numeric_limits<double>::infinity();
  double t_far  = std::numeric_limits<double>::infinity();
  for (const Eigen::Vector4d &plane : m_scaledPlanes)
  {
    const double dist  = plane(0)*orig.x() + plane(1)*orig.y() + plane(2)*orig.z() + plane(3);
    const double denom = plane(0)*dr.x() + plane(1)*dr.y() + plane(2)*dr.z();
    if (std::fabs(denom) < ZERO)
    {
      if (dist > 0.0) return false;  // parallel to this face, on its outer side
      continue;
    }
    const double t = -dist / denom;
    if (denom < 0.0)
      t_near = std::max(t_near, t);
    else
      t_far = std::min(t_far, t);
    if (t_near > t_far) return false;
  }
  if (t_far <= 0.0) return false;  // the hull is behind the origin

  if (intersections)
  {
    // From outside the ray meets the hull twice, unless it only grazes it; from inside, once.
    const bool enters = t_near > 0.0 && (t_far - t_near) > ZERO;
    if (enters)
      intersections->push_back(origin + dir * t_near);
    if (!enters || count != 1)
      intersections->push_back(origin + dir * t_far);
  }
  return true;
}
double ConvexMesh::computeVolume() const
{
  // Same as original
  double volume = 0.0;
  for (size_t i = 0; i < m_triangles.size() / 3; ++i)
  {
    const tf2::Vector3 &v1 = m_vertices[m_triangles[3*i+0]];
    const tf2::Vector3 &v2 = m_vertices[m_triangles[3*i+1]];
    const tf2::Vector3 &v3 = m_vertices[m_triangles[3*i+2]];
    volume += v1.x()*v2.y()*v3.z() + v2.x()*v3.y()*v1.z() + v3.x()*v1.y()*v2.z()
            - v1.x()*v3.y()*v2.z() - v2.x()*v1.y()*v3.z() - v3.x()*v2.y()*v1.z();
  }
  return std::fabs(volume) / 6.0;
}
void ConvexMesh::computeBoundingSphere(BoundingSphere &sphere) const
{
  sphere.center = m_center;
  sphere.radius = m_radiusB;
}
void ConvexMesh::useDimensions(const shapes::Shape *shape)
{
  // same approach as original for convex hull
  const shapes::Mesh *mesh = static_cast<const shapes::Mesh*>(shape);
  double maxX=-1e30, maxY=-1e30, maxZ=-1e30;
  double minX=1e30,  minY=1e30,  minZ=1e30;
  for (unsigned int i = 0; i < mesh->vertexCount; ++i)
  {
    double vx = mesh->vertices[3*i];
    double vy = mesh->vertices[3*i+1];
    double vz = mesh->vertices[3*i+2];
    if (vx > maxX) maxX = vx;
    if (vy > maxY) maxY = vy;
    if (vz > maxZ) maxZ = vz;
    if (vx < minX) minX = vx;
    if (vy < minY) minY = vy;
    if (vz < minZ) minZ = vz;
  }
  shapes::Box *box_shape = new shapes::Box(maxX-minX, maxY-minY, maxZ-minZ);
  m_boundingBox.setDimensions(box_shape);
  delete box_shape;

  m_boxOffset.setValue((minX+maxX)/2.0, (minY+maxY)/2.0, (minZ+maxZ)/2.0);
  m_planes.clear(); 
  m_triangles.clear(); 
  m_vertices.clear();
  m_vertexNormals.clear();
  m_meshRadiusB = 0.0; 
  m_meshCenter.setValue(0,0,0);

  btVector3 *vertices = new btVector3[mesh->vertexCount];
  for (unsigned int i=0; i<mesh->vertexCount; i++)
    vertices[i].setValue(
      static_cast<btScalar>(mesh->vertices[3*i]),
      static_cast<btScalar>(mesh->vertices[3*i+1]),
      static_cast<btScalar>(mesh->vertices[3*i+2]));

  HullDesc hd(QF_TRIANGLES, mesh->vertexCount, vertices);
  HullResult hr;
  HullLibrary hl;
  if (hl.CreateConvexHull(hd, hr) == QE_OK)
  {
    tf2::Vector3 sum(0,0,0);
    m_vertices.reserve(hr.m_OutputVertices.size());
    for (int j = 0; j < hr.m_OutputVertices.size(); ++j)
    {
      tf2::Vector3 v(hr.m_OutputVertices[j][0],
                     hr.m_OutputVertices[j][1],
                     hr.m_OutputVertices[j][2]);
      m_vertices.push_back(v);
      sum += v;
    }
    m_meshCenter = sum / (double)hr.m_OutputVertices.size();
    for (auto &v : m_vertices)
    {
      double dist = v.distance2(m_meshCenter);
      if (dist > m_meshRadiusB) m_meshRadiusB = dist;
    }
    m_meshRadiusB = std::sqrt(m_meshRadiusB);

    m_triangles.reserve(hr.m_Indices.size());
    for (unsigned int j=0; j<hr.mNumFaces; j++)
    {
      unsigned int i0 = hr.m_Indices[j*3+0];
      unsigned int i1 = hr.m_Indices[j*3+1];
      unsigned int i2 = hr.m_Indices[j*3+2];
      tf2::Vector3 p1(hr.m_OutputVertices[i0][0],
                      hr.m_OutputVertices[i0][1],
                      hr.m_OutputVertices[i0][2]);
      tf2::Vector3 p2(hr.m_OutputVertices[i1][0],
                      hr.m_OutputVertices[i1][1],
                      hr.m_OutputVertices[i1][2]);
      tf2::Vector3 p3(hr.m_OutputVertices[i2][0],
                      hr.m_OutputVertices[i2][1],
                      hr.m_OutputVertices[i2][2]);
      tf2::Vector3 e1 = p2 - p1; e1.normalize();
      tf2::Vector3 e2 = p3 - p1; e2.normalize();
      tf2::Vector3 planeNormal = e1.cross(e2);
      if (planeNormal.length2() > 1e-6)
      {
        planeNormal.normalize();
        Eigen::Vector4d planeEq;
        planeEq << planeNormal.x(), planeNormal.y(), planeNormal.z(), -planeNormal.dot(p1);
        unsigned int behindPlane = countVerticesBehindPlane(planeEq);
        if (behindPlane > 0)
        {
          Eigen::Vector4d planeEq2 = -planeEq;
          unsigned int behindPlane2 = countVerticesBehindPlane(planeEq2);
          if (behindPlane2 < behindPlane) planeEq = planeEq2;
        }
        m_planes.push_back(planeEq);
        m_triangles.push_back(i0);
        m_triangles.push_back(i1);
        m_triangles.push_back(i2);
      }
    }

    // Only for drawing the padded hull; the filter itself pads the planes, not the vertices.
    m_vertexNormals.assign(m_vertices.size(), tf2::Vector3(0.0, 0.0, 0.0));
    for (size_t i = 0; i < m_planes.size(); ++i)
    {
      const tf2::Vector3 normal(m_planes[i](0), m_planes[i](1), m_planes[i](2));
      for (size_t k = 0; k < 3; ++k)
        m_vertexNormals[m_triangles[3 * i + k]] += normal;
    }
    for (auto &normal : m_vertexNormals)
      if (normal.length2() > 0.0) normal.normalize();
  }
  else
  {
    std::cerr << "[ConvexMesh::useDimensions] Unable to compute convex hull!" << std::endl;
  }
  hl.ReleaseResult(hr);
  delete[] vertices;
  updateScaledGeometry();
}
void ConvexMesh::updateScaledGeometry()
{
  // Scaling about the centroid c takes face n.x + d = 0 to n.x + s*d + (s - 1)*n.c = 0, and the
  // padding then pushes it out along its unit normal n.
  m_scaledPlanes.resize(m_planes.size());
  for (size_t i = 0; i < m_planes.size(); ++i)
  {
    const Eigen::Vector4d &plane = m_planes[i];
    const double n_dot_c = plane(0) * m_meshCenter.x() + plane(1) * m_meshCenter.y() +
      plane(2) * m_meshCenter.z();
    m_scaledPlanes[i] = plane;
    m_scaledPlanes[i](3) = m_scale * plane(3) + (m_scale - 1.0) * n_dot_c - m_padding;
  }

  m_scaledVertices.resize(m_vertices.size());
  for (size_t i = 0; i < m_vertices.size(); ++i)
    m_scaledVertices[i] = m_meshCenter + (m_vertices[i] - m_meshCenter) * m_scale +
      m_vertexNormals[i] * m_padding;

  m_boundingBox.setScale(m_scale, m_scale, m_scale);
  m_boundingBox.setPadding(m_padding, m_padding, m_padding);
}
void ConvexMesh::updateInternalData()
{
  // The hull scales about its centroid, so the box around it moves as well as grows.
  tf2::Transform pose = m_pose;
  pose.setOrigin(m_pose * (m_meshCenter + (m_boxOffset - m_meshCenter) * m_scale));
  m_boundingBox.setPose(pose);
  m_iPose      = m_pose.inverse();
  m_center     = m_pose * m_meshCenter;
  m_radiusB    = m_meshRadiusB * m_scale + m_padding;
  m_radiusBSqr = m_radiusB * m_radiusB;
}
bool ConvexMesh::isPointInsidePlanes(const tf2::Vector3 &point) const
{
  for (auto &plane : m_scaledPlanes)
  {
    double dist = plane(0)*point.x() + plane(1)*point.y() + plane(2)*point.z() + plane(3);
    if (dist > 0.0) return false;
  }
  return true;
}
unsigned int ConvexMesh::countVerticesBehindPlane(const Eigen::Vector4d &planeNormal) const
{
  unsigned int result = 0;
  for (auto &v : m_vertices)
  {
    double dist = planeNormal(0)*v.x() + planeNormal(1)*v.y() + planeNormal(2)*v.z() + planeNormal(3);
    if (dist > 1e-6) result++;
  }
  return result;
}

} // namespace bodies
} // namespace robot_self_filter
