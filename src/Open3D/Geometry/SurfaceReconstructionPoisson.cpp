#include "Open3D/Geometry/IntersectionTest.h"
#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

#include <Eigen/Dense>
#include <iostream>
#include <list>

#include "poisson/PreProcessor.h"

#define DATA_DEGREE 0  // The order of the B-Spline used to splat in data for color interpolation
#define WEIGHT_DEGREE \
    2  // The order of the B-Spline used to splat in the weights for density estimation
#define NORMAL_DEGREE \
    2  // The order of the B-Spline used to splat in the normals for constructing the Laplacian
       // constraints
#define DEFAULT_FEM_DEGREE 1                   // The default finite-element degree
#define DEFAULT_FEM_BOUNDARY BOUNDARY_NEUMANN  // The default finite-element boundary type
#define DIMENSION 3                            // The dimension of the system

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "poisson/CmdLineParser.h"
#include "poisson/FEMTree.h"
#include "poisson/MyMiscellany.h"
#include "poisson/PPolynomial.h"
#include "poisson/Ply.h"
#include "poisson/PointStreamData.h"

namespace open3d {
namespace geometry {

namespace poisson {
typedef Eigen::Matrix<double, 6, 1> Open3DData;
// typedef Eigen::Matrix<double, 7, 1> Open3DData;

class Open3DPointStream : public InputPointStreamWithData<double, DIMENSION, Open3DData> {
public:
    // Open3DPointStream() {}
    Open3DPointStream(const open3d::geometry::PointCloud* pcd)
        : pcd_(pcd), xform_(nullptr), current_(0) {
        open3d::utility::LogDebug("O3DPointStream pcd.points {}", pcd_->points_.size());
        open3d::utility::LogDebug("O3DPointStream pcd.colors {}", pcd_->colors_.size());
        open3d::utility::LogDebug("O3DPointStream pcd.normals {}", pcd_->normals_.size());
    }
    void reset(void) { current_ = 0; }
    bool nextPoint(Point<double, 3>& p, Open3DData& d) {
        if (current_ >= pcd_->points_.size()) {
            return false;
        }
        p.coords[0] = pcd_->points_[current_](0);
        p.coords[1] = pcd_->points_[current_](1);
        p.coords[2] = pcd_->points_[current_](2);

        if (xform_ != nullptr) {
            p = (*xform_) * p;
        }

        if (pcd_->HasNormals()) {
            d(0) = pcd_->normals_[current_](0);
            d(1) = pcd_->normals_[current_](1);
            d(2) = pcd_->normals_[current_](2);
        } else {
            d(0) = 0;
            d(1) = 0;
            d(2) = 0;
        }

        if (pcd_->HasColors()) {
            d(3) = pcd_->colors_[current_](0);
            d(4) = pcd_->colors_[current_](1);
            d(5) = pcd_->colors_[current_](2);
        } else {
            d(3) = 0;
            d(4) = 0;
            d(5) = 0;
        }

        // d(6) = 0;

        current_++;
        return true;
    }

public:
    const open3d::geometry::PointCloud* pcd_;
    XForm<double, 4>* xform_;
    size_t current_;
};

class Open3DVertex {
public:
    typedef double Real;

    Open3DVertex(Point<double, 3> point) : point(point) {}
    Open3DVertex() {}
    virtual ~Open3DVertex() {}

    Open3DVertex& operator*=(double s) {
        point *= s;
        data *= s;
        return *this;
    }

    Open3DVertex& operator+=(const Open3DVertex& p) {
        point += p.point;
        data += p.data;
        return *this;
    }

    Open3DVertex& operator/=(double s) {
        point /= s;
        data /= s;
        return *this;
    }

public:
    Point<double, 3> point;
    Eigen::Matrix<double, 7, 1> data;
};

double Weight(double v, double start, double end) {
    v = (v - start) / (end - start);
    if (v < 0)
        return 1.;
    else if (v > 1)
        return 0.;
    else {
        return 2. * v * v * v - 3. * v * v + 1.;
    }
}

template <unsigned int Dim, class Real>
struct FEMTreeProfiler {
    FEMTree<Dim, Real>& tree;
    double t;

    FEMTreeProfiler(FEMTree<Dim, Real>& t) : tree(t) { ; }
    void start(void) { t = Time(), FEMTree<Dim, Real>::ResetLocalMemoryUsage(); }
    void print(const char* header) const {
        FEMTree<Dim, Real>::MemoryUsage();
        if (header)
            printf("%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n", header, Time() - t,
                   FEMTree<Dim, Real>::LocalMemoryUsage(), FEMTree<Dim, Real>::MaxMemoryUsage(),
                   MemoryInfo::PeakMemoryUsageMB());
        else
            printf("%9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n", Time() - t,
                   FEMTree<Dim, Real>::LocalMemoryUsage(), FEMTree<Dim, Real>::MaxMemoryUsage(),
                   MemoryInfo::PeakMemoryUsageMB());
    }
    void dumpOutput(const char* header) const {
        FEMTree<Dim, Real>::MemoryUsage();
        // if (header)
        //     messageWriter("%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n", header,
        //                   Time() - t, FEMTree<Dim, Real>::LocalMemoryUsage(),
        //                   FEMTree<Dim, Real>::MaxMemoryUsage(),
        //                   MemoryInfo::PeakMemoryUsageMB());
        // else
        //     messageWriter("%9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n", Time() - t,
        //                   FEMTree<Dim, Real>::LocalMemoryUsage(),
        //                   FEMTree<Dim, Real>::MaxMemoryUsage(),
        //                   MemoryInfo::PeakMemoryUsageMB());
    }
    void dumpOutput2(std::vector<std::string>& comments, const char* header) const {
        FEMTree<Dim, Real>::MemoryUsage();
        // if (header)
        //     messageWriter(comments, "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n",
        //     header,
        //                   Time() - t, FEMTree<Dim, Real>::LocalMemoryUsage(),
        //                   FEMTree<Dim, Real>::MaxMemoryUsage(),
        //                   MemoryInfo::PeakMemoryUsageMB());
        // else
        //     messageWriter(comments, "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n",
        //     Time() - t,
        //                   FEMTree<Dim, Real>::LocalMemoryUsage(),
        //                   FEMTree<Dim, Real>::MaxMemoryUsage(),
        //                   MemoryInfo::PeakMemoryUsageMB());
    }
};

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetBoundingBoxXForm(Point<Real, Dim> min,
                                         Point<Real, Dim> max,
                                         Real scaleFactor) {
    Point<Real, Dim> center = (max + min) / 2;
    Real scale = max[0] - min[0];
    for (int d = 1; d < Dim; d++) scale = std::max<Real>(scale, max[d] - min[d]);
    scale *= scaleFactor;
    for (int i = 0; i < Dim; i++) center[i] -= scale / 2;
    XForm<Real, Dim + 1> tXForm = XForm<Real, Dim + 1>::Identity(),
                         sXForm = XForm<Real, Dim + 1>::Identity();
    for (int i = 0; i < Dim; i++) sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
    return sXForm * tXForm;
}
template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetBoundingBoxXForm(
        Point<Real, Dim> min, Point<Real, Dim> max, Real width, Real scaleFactor, int& depth) {
    // Get the target resolution (along the largest dimension)
    Real resolution = (max[0] - min[0]) / width;
    for (int d = 1; d < Dim; d++)
        resolution = std::max<Real>(resolution, (max[d] - min[d]) / width);
    resolution *= scaleFactor;
    depth = 0;
    while ((1 << depth) < resolution) depth++;

    Point<Real, Dim> center = (max + min) / 2;
    Real scale = (1 << depth) * width;

    for (int i = 0; i < Dim; i++) center[i] -= scale / 2;
    XForm<Real, Dim + 1> tXForm = XForm<Real, Dim + 1>::Identity(),
                         sXForm = XForm<Real, Dim + 1>::Identity();
    for (int i = 0; i < Dim; i++) sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
    return sXForm * tXForm;
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetPointXForm(InputPointStream<Real, Dim>& stream,
                                   Real width,
                                   Real scaleFactor,
                                   int& depth) {
    Point<Real, Dim> min, max;
    stream.boundingBox(min, max);
    return GetBoundingBoxXForm(min, max, width, scaleFactor, depth);
}
template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetPointXForm(InputPointStream<Real, Dim>& stream, Real scaleFactor) {
    Point<Real, Dim> min, max;
    stream.boundingBox(min, max);
    return GetBoundingBoxXForm(min, max, scaleFactor);
}

template <unsigned int Dim, typename Real>
struct ConstraintDual {
    Real target, weight;
    ConstraintDual(Real t, Real w) : target(t), weight(w) {}
    CumulativeDerivativeValues<Real, Dim, 0> operator()(const Point<Real, Dim>& p) const {
        return CumulativeDerivativeValues<Real, Dim, 0>(target * weight);
    };
};
template <unsigned int Dim, typename Real>
struct SystemDual {
    Real weight;
    SystemDual(Real w) : weight(w) {}
    CumulativeDerivativeValues<Real, Dim, 0> operator()(
            const Point<Real, Dim>& p,
            const CumulativeDerivativeValues<Real, Dim, 0>& dValues) const {
        return dValues * weight;
    };
    CumulativeDerivativeValues<double, Dim, 0> operator()(
            const Point<Real, Dim>& p,
            const CumulativeDerivativeValues<double, Dim, 0>& dValues) const {
        return dValues * weight;
    };
};
template <unsigned int Dim>
struct SystemDual<Dim, double> {
    typedef double Real;
    Real weight;
    SystemDual(Real w) : weight(w) {}
    CumulativeDerivativeValues<Real, Dim, 0> operator()(
            const Point<Real, Dim>& p,
            const CumulativeDerivativeValues<Real, Dim, 0>& dValues) const {
        return dValues * weight;
    };
};

template <typename Vertex,
          typename Real,
          typename SetVertexFunction,
          unsigned int... FEMSigs,
          typename... SampleData>
void ExtractMesh(
        UIntPack<FEMSigs...>,
        std::tuple<SampleData...>,
        FEMTree<sizeof...(FEMSigs), Real>& tree,
        const DenseNodeData<Real, UIntPack<FEMSigs...>>& solution,
        Real isoValue,
        const std::vector<typename FEMTree<sizeof...(FEMSigs), Real>::PointSample>* samples,
        std::vector<Open3DData>* sampleData,
        const typename FEMTree<sizeof...(FEMSigs), Real>::template DensityEstimator<WEIGHT_DEGREE>*
                density,
        const SetVertexFunction& SetVertex,
        XForm<Real, sizeof...(FEMSigs) + 1> iXForm,
        std::shared_ptr<open3d::geometry::TriangleMesh>& out_mesh) {
    static const int Dim = sizeof...(FEMSigs);
    typedef UIntPack<FEMSigs...> Sigs;
    static const unsigned int DataSig = FEMDegreeAndBType<DATA_DEGREE, BOUNDARY_FREE>::Signature;
    typedef typename FEMTree<Dim, Real>::template DensityEstimator<WEIGHT_DEGREE> DensityEstimator;

    FEMTreeProfiler<Dim, Real> profiler(tree);

    CoredMeshData<Vertex, node_index_type>* mesh;
    mesh = new CoredVectorMeshData<Vertex, node_index_type>();

    float datax = 32.f;
    bool linear_fit = true;
    bool non_manifold = true;
    bool polygon_mesh = false;

    profiler.start();
    typename IsoSurfaceExtractor<Dim, Real, Vertex>::IsoStats isoStats;
    if (sampleData) {
        SparseNodeData<ProjectiveData<Open3DData, Real>, IsotropicUIntPack<Dim, DataSig>>
                _sampleData = tree.template setMultiDepthDataField<DataSig, false>(
                        *samples, *sampleData, (DensityEstimator*)NULL);
        for (const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* n =
                     tree.tree().nextNode();
             n; n = tree.tree().nextNode(n)) {
            ProjectiveData<Open3DData, Real>* clr = _sampleData(n);
            if (clr) (*clr) *= (Real)pow(datax, tree.depth(n));
        }
        isoStats = IsoSurfaceExtractor<Dim, Real, Vertex>::template Extract<Open3DData>(
                Sigs(), UIntPack<WEIGHT_DEGREE>(), UIntPack<DataSig>(), tree, density, &_sampleData,
                solution, isoValue, *mesh, SetVertex, !linear_fit, !non_manifold, polygon_mesh,
                false);
    }
    // #if defined(__GNUC__) && __GNUC__ < 5
    // #warning "you've got me gcc version<5"
    //     else
    //         isoStats = IsoSurfaceExtractor<Dim, Real, Vertex>::template
    //         Extract<Open3DData>(
    //                 Sigs(), UIntPack<WEIGHT_DEGREE>(), UIntPack<DataSig>(), tree, density,
    //                 (SparseNodeData<ProjectiveData<Open3DData, Real>,
    //                                 IsotropicUIntPack<Dim, DataSig>>*)NULL,
    //                 solution, isoValue, *mesh, SetVertex, !linear_fit, !non_manifold,
    //                 polygon_mesh, false);
    // #else   // !__GNUC__ || __GNUC__ >=5
    else
        isoStats = IsoSurfaceExtractor<Dim, Real, Vertex>::template Extract<Open3DData>(
                Sigs(), UIntPack<WEIGHT_DEGREE>(), UIntPack<DataSig>(), tree, density, NULL,
                solution, isoValue, *mesh, SetVertex, !linear_fit, !non_manifold, polygon_mesh,
                false);
    // #endif  // __GNUC__ || __GNUC__ < 4
    // messageWriter("Vertices / Polygons: %llu / %llu\n",
    //               (unsigned long long)(mesh->outOfCorePointCount() +
    //               mesh->inCorePoints.size()), (unsigned long long)mesh->polygonCount());
    std::string isoStatsString = isoStats.toString() + std::string("\n");
    // messageWriter(isoStatsString.c_str());
    // if (polygon_mesh)
    //     profiler.dumpOutput2(comments, "#         Got polygons:");
    // else
    //     profiler.dumpOutput2(comments, "#        Got triangles:");

    mesh->resetIterator();
    for (size_t vidx = 0; vidx < mesh->outOfCorePointCount(); ++vidx) {
        Vertex v;
        mesh->nextOutOfCorePoint(v);
        out_mesh->vertices_.push_back(Eigen::Vector3d(v.point[0], v.point[1], v.point[2]));
        out_mesh->vertex_normals_.push_back(Eigen::Vector3d(v.data(0), v.data(1), v.data(2)));
        out_mesh->vertex_colors_.push_back(Eigen::Vector3d(v.data(3), v.data(4), v.data(5)));
    }
    for (size_t tidx = 0; tidx < mesh->polygonCount(); ++tidx) {
        std::vector<CoredVertexIndex<node_index_type>> triangle;
        mesh->nextPolygon(triangle);
        if (triangle.size() != 3) {
            open3d::utility::LogError("got polygon");
        } else {
            out_mesh->triangles_.push_back(
                    Eigen::Vector3i(triangle[0].idx, triangle[1].idx, triangle[2].idx));
        }
    }

    delete mesh;
}

template <class Real, typename... SampleData, unsigned int... FEMSigs>
void Execute(const open3d::geometry::PointCloud& pcd,
             std::shared_ptr<open3d::geometry::TriangleMesh>& out_mesh,
             UIntPack<FEMSigs...>) {
    static const int Dim = sizeof...(FEMSigs);
    typedef UIntPack<FEMSigs...> Sigs;
    typedef UIntPack<FEMSignature<FEMSigs>::Degree...> Degrees;
    typedef UIntPack<
            FEMDegreeAndBType<NORMAL_DEGREE, DerivativeBoundary<FEMSignature<FEMSigs>::BType,
                                                                1>::BType>::Signature...>
            NormalSigs;
    static const unsigned int DataSig = FEMDegreeAndBType<DATA_DEGREE, BOUNDARY_FREE>::Signature;
    typedef typename FEMTree<Dim, Real>::template DensityEstimator<WEIGHT_DEGREE> DensityEstimator;
    typedef typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 0> InterpolationInfo;
    // std::vector<std::string> comments;
    // messageWriter(comments,
    // "*************************************************************\n");
    // messageWriter(comments,
    // "*************************************************************\n");
    // messageWriter(comments, "** Running Screened Poisson Reconstruction (Version %s) **\n",
    //               VERSION);
    // messageWriter(comments,
    // "*************************************************************\n");
    // messageWriter(comments,
    // "*************************************************************\n"); if (!Threads.set)
    // messageWriter(comments, "Running with %d threads\n", Threads.value);

    XForm<Real, Dim + 1> xForm, iXForm;
    xForm = XForm<Real, Dim + 1>::Identity();

    float datax = 32.f;
    float width = 0;
    int depth = 8;
    float base_depth = 0;
    float base_v_cycles = 1;
    float scale = 1.1;
    float confidence = 0.f;
    float point_weight = 2 * DEFAULT_FEM_DEGREE;
    float confidence_bias = 0.f;
    float samples_per_node = 1.5f;
    float cg_solver_accuracy = 1e-3;
    float full_depth = 5.f;
    int iters = 8;
    bool exact_interpolation = false;
    bool Density = true;
    bool Colors = pcd.HasColors();
    bool Normals = pcd.HasNormals();

    double startTime = Time();
    Real isoValue = 0;

    FEMTree<Dim, Real> tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
    FEMTreeProfiler<Dim, Real> profiler(tree);

    size_t pointCount;

    Real pointWeightSum;
    std::vector<typename FEMTree<Dim, Real>::PointSample>* samples =
            new std::vector<typename FEMTree<Dim, Real>::PointSample>();
    std::vector<Open3DData> sampleData;
    DensityEstimator* density = NULL;
    SparseNodeData<Point<Real, Dim>, NormalSigs>* normalInfo = NULL;
    Real targetValue = (Real)0.5;

    // Read in the samples (and color data)
    {
        Open3DPointStream pointStream(&pcd);

        if (width > 0)
            xForm = GetPointXForm<Real, Dim>(pointStream, width, (Real)(scale > 0 ? scale : 1.),
                                             depth) *
                    xForm;
        else
            xForm = scale > 0 ? GetPointXForm<Real, Dim>(pointStream, (Real)scale) * xForm : xForm;

        pointStream.xform_ = &xForm;

        {
            auto ProcessDataWithConfidence = [&](const Point<Real, Dim>& p, Open3DData& d) {
                // Real l = (Real)Length(d.template data<0>());
                Real l = (Real)d.head<3>().norm();
                if (!l || l != l) return (Real)-1.;
                return (Real)pow(l, confidence);
            };
            auto ProcessData = [](const Point<Real, Dim>& p, Open3DData& d) {
                // Real l = (Real)Length(d.template data<0>());
                Real l = (Real)d.head<3>().norm();
                if (!l || l != l) return (Real)-1.;
                d.head<3>() /= l;
                return (Real)1.;
            };
            if (confidence > 0)
                pointCount = FEMTreeInitializer<Dim, Real>::template Initialize<Open3DData>(
                        tree.spaceRoot(), pointStream, depth, *samples, sampleData, true,
                        tree.nodeAllocators[0], tree.initializer(), ProcessDataWithConfidence);
            else
                pointCount = FEMTreeInitializer<Dim, Real>::template Initialize<Open3DData>(
                        tree.spaceRoot(), pointStream, depth, *samples, sampleData, true,
                        tree.nodeAllocators[0], tree.initializer(), ProcessData);
        }
        iXForm = xForm.inverse();
        printf("[[%f,%f,%f], [%f,%f,%f], [%f,%f,%f]]\n", xForm.coords[0][0], xForm.coords[0][1],
               xForm.coords[0][2], xForm.coords[1][0], xForm.coords[1][1], xForm.coords[1][2],
               xForm.coords[2][0], xForm.coords[2][1], xForm.coords[2][2]);
        printf("[[%f,%f,%f], [%f,%f,%f], [%f,%f,%f]]\n", iXForm.coords[0][0], iXForm.coords[0][1],
               iXForm.coords[0][2], iXForm.coords[1][0], iXForm.coords[1][1], iXForm.coords[1][2],
               iXForm.coords[2][0], iXForm.coords[2][1], iXForm.coords[2][2]);

        // messageWriter("Input Points / Samples: %llu / %llu\n", (unsigned long
        // long)pointCount,
        //               (unsigned long long)samples->size());
    }
    // end of read pcd

    // int kernelDepth = KernelDepth.set ? KernelDepth.value : Depth.value - 2;
    int kernelDepth = depth - 2;
    if (kernelDepth > depth) {
        // TODO warning
        kernelDepth = depth;
    }

    DenseNodeData<Real, Sigs> solution;
    {
        DenseNodeData<Real, Sigs> constraints;
        InterpolationInfo* iInfo = NULL;
        int solveDepth = depth;

        tree.resetNodeIndices();

        // Get the kernel density estimator
        {
            profiler.start();
            density = tree.template setDensityEstimator<WEIGHT_DEGREE>(*samples, kernelDepth,
                                                                       samples_per_node, 1);
            // profiler.dumpOutput2(comments, "#   Got kernel density:");
        }

        // Transform the Hermite samples into a vector field
        {
            profiler.start();
            normalInfo = new SparseNodeData<Point<Real, Dim>, NormalSigs>();
            std::function<bool(Open3DData, Point<Real, Dim>&)> ConversionFunction =
                    [](Open3DData in, Point<Real, Dim>& out) {
                        // Point<Real, Dim> n = in.template data<0>();
                        Point<Real, Dim> n(in(0), in(1), in(2));
                        Real l = (Real)Length(n);
                        // It is possible that the samples have non-zero normals but there are
                        // two co-located samples with negative normals...
                        if (!l) return false;
                        out = n / l;
                        return true;
                    };
            std::function<bool(Open3DData, Point<Real, Dim>&, Real&)> ConversionAndBiasFunction =
                    [&](Open3DData in, Point<Real, Dim>& out, Real& bias) {
                        // Point<Real, Dim> n = in.template data<0>();
                        Point<Real, Dim> n(in(0), in(1), in(2));
                        Real l = (Real)Length(n);
                        // It is possible that the samples have non-zero normals but
                        // there are two co-located samples with negative normals...
                        if (!l) return false;
                        out = n / l;
                        bias = (Real)(log(l) * confidence_bias / log(1 << (Dim - 1)));
                        return true;
                    };
            if (confidence_bias > 0)
                *normalInfo = tree.setDataField(NormalSigs(), *samples, sampleData, density,
                                                pointWeightSum, ConversionAndBiasFunction);
            else
                *normalInfo = tree.setDataField(NormalSigs(), *samples, sampleData, density,
                                                pointWeightSum, ConversionFunction);
            ThreadPool::Parallel_for(0, normalInfo->size(), [&](unsigned int, size_t i) {
                (*normalInfo)[i] *= (Real)-1.;
            });
            // profiler.dumpOutput2(comments, "#     Got normal field:");
            // messageWriter("Point weight / Estimated Area: %g / %g\n", pointWeightSum,
            //               pointCount * pointWeightSum);
        }

        if (!Density) delete density, density = NULL;

        // Trim the tree and prepare for multigrid
        {
            profiler.start();
            constexpr int MAX_DEGREE =
                    NORMAL_DEGREE > Degrees::Max() ? NORMAL_DEGREE : Degrees::Max();
            tree.template finalizeForMultigrid<MAX_DEGREE>(
                    full_depth,
                    typename FEMTree<Dim, Real>::template HasNormalDataFunctor<NormalSigs>(
                            *normalInfo),
                    normalInfo, density);
            // profiler.dumpOutput2(comments, "#       Finalized tree:");
        }
        // Add the FEM constraints
        {
            profiler.start();
            constraints = tree.initDenseNodeData(Sigs());
            typename FEMIntegrator::template Constraint<Sigs, IsotropicUIntPack<Dim, 1>, NormalSigs,
                                                        IsotropicUIntPack<Dim, 0>, Dim>
                    F;
            unsigned int derivatives2[Dim];
            for (int d = 0; d < Dim; d++) derivatives2[d] = 0;
            typedef IsotropicUIntPack<Dim, 1> Derivatives1;
            typedef IsotropicUIntPack<Dim, 0> Derivatives2;
            for (int d = 0; d < Dim; d++) {
                unsigned int derivatives1[Dim];
                for (int dd = 0; dd < Dim; dd++) derivatives1[dd] = dd == d ? 1 : 0;
                F.weights[d][TensorDerivatives<Derivatives1>::Index(derivatives1)]
                         [TensorDerivatives<Derivatives2>::Index(derivatives2)] = 1;
            }
            tree.addFEMConstraints(F, *normalInfo, constraints, solveDepth);
            // profiler.dumpOutput2(comments, "#  Set FEM constraints:");
        }

        // Free up the normal info
        delete normalInfo, normalInfo = NULL;

        // Add the interpolation constraints
        if (point_weight > 0) {
            profiler.start();
            if (exact_interpolation)
                iInfo = FEMTree<Dim, Real>::template InitializeExactPointInterpolationInfo<Real, 0>(
                        tree, *samples,
                        ConstraintDual<Dim, Real>(targetValue, (Real)point_weight * pointWeightSum),
                        SystemDual<Dim, Real>((Real)point_weight * pointWeightSum), true, false);
            else
                iInfo = FEMTree<Dim, Real>::template InitializeApproximatePointInterpolationInfo<
                        Real, 0>(
                        tree, *samples,
                        ConstraintDual<Dim, Real>(targetValue, (Real)point_weight * pointWeightSum),
                        SystemDual<Dim, Real>((Real)point_weight * pointWeightSum), true, 1);
            tree.addInterpolationConstraints(constraints, solveDepth, *iInfo);
            // profiler.dumpOutput2(comments, "#Set point constraints:");
        }

        // messageWriter("Leaf Nodes / Active Nodes / Ghost Nodes: %llu / %llu / %llu\n",
        //               (unsigned long long)tree.leaves(), (unsigned long long)tree.nodes(),
        //               (unsigned long long)tree.ghostNodes());
        // messageWriter("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));

        // Solve the linear system
        {
            profiler.start();
            typename FEMTree<Dim, Real>::SolverInfo sInfo;
            sInfo.cgDepth = 0, sInfo.cascadic = true, sInfo.vCycles = 1, sInfo.iters = iters,
            sInfo.cgAccuracy = cg_solver_accuracy, sInfo.verbose = true, sInfo.showResidual = true,
            sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE, sInfo.sliceBlockSize = 1;
            sInfo.baseDepth = base_depth, sInfo.baseVCycles = base_v_cycles;
            typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim, 1>> F({0., 1.});
            solution = tree.solveSystem(Sigs(), F, constraints, solveDepth, sInfo, iInfo);
            // profiler.dumpOutput2(comments, "# Linear system solved:");
            if (iInfo) delete iInfo, iInfo = NULL;
        }
    }

    {
        profiler.start();
        double valueSum = 0, weightSum = 0;
        typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 0> evaluator(&tree,
                                                                                        solution);
        std::vector<double> valueSums(ThreadPool::NumThreads(), 0),
                weightSums(ThreadPool::NumThreads(), 0);
        ThreadPool::Parallel_for(0, samples->size(), [&](unsigned int thread, size_t j) {
            ProjectiveData<Point<Real, Dim>, Real>& sample = (*samples)[j].sample;
            Real w = sample.weight;
            if (w > 0)
                weightSums[thread] += w,
                        valueSums[thread] += evaluator.values(sample.data / sample.weight, thread,
                                                              (*samples)[j].node)[0] *
                                             w;
        });
        for (size_t t = 0; t < valueSums.size(); t++)
            valueSum += valueSums[t], weightSum += weightSums[t];
        isoValue = (Real)(valueSum / weightSum);
        if (datax <= 0 || (!Colors && !Normals)) delete samples, samples = NULL;
        profiler.dumpOutput("Got average:");
        // messageWriter("Iso-Value: %e = %g / %g\n", isoValue, valueSum, weightSum);
    }

    auto SetVertex = [](Open3DVertex& v, Point<Real, Dim> p, Real w, Open3DData d) {
        v.point = p;
        v.data(0) = d(0);
        v.data(1) = d(1);
        v.data(2) = d(2);
        v.data(3) = d(3);
        v.data(4) = d(4);
        v.data(5) = d(5);
        v.data(6) = w;
    };
    ExtractMesh<Open3DVertex>(UIntPack<FEMSigs...>(), std::tuple<SampleData...>(), tree, solution,
                              isoValue, samples, &sampleData, density, SetVertex, iXForm, out_mesh);

    if (density) delete density, density = NULL;
    // messageWriter(comments, "#          Total Solve: %9.1f (s), %9.1f (MB)\n", Time() -
    // startTime,
    //               FEMTree<Dim, Real>::MaxMemoryUsage());
}

}  // namespace poisson

std::shared_ptr<TriangleMesh> TriangleMesh::CreateFromPointCloudPoisson(const PointCloud& pcd) {
    static const BoundaryType BType = DEFAULT_FEM_BOUNDARY;
    typedef IsotropicUIntPack<DIMENSION, FEMDegreeAndBType</* Degree */ 1, BType>::Signature>
            FEMSigs;

#ifdef _OPENMP
    ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP,
                     std::thread::hardware_concurrency());
#else
    ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL,
                     std::thread::hardware_concurrency());
#endif

    auto mesh = std::make_shared<TriangleMesh>();
    poisson::Execute<double>(pcd, mesh, FEMSigs());

    ThreadPool::Terminate();

    return mesh;
}

}  // namespace geometry
}  // namespace open3d
