#include <cfloat>
#include <random>
#include <chrono>

#include "vec.h"
#include "mesh.h"
#include "wavefront.h"
#include "orbiter.h"

#include "ray.h"

#include "image.h"
#include "image_io.h"
#include "image_hdr.h"


Vector normal( const Hit& hit, const TriangleData& triangle )
{
    return normalize((1 - hit.u - hit.v) * Vector(triangle.na) + hit.u * Vector(triangle.nb) + hit.v * Vector(triangle.nc));
}

Point point( const Hit& hit, const TriangleData& triangle )
{
    return (1 - hit.u - hit.v) * Point(triangle.a) + hit.u * Point(triangle.b) + hit.v * Point(triangle.c);
}


Point point( const Hit& hit, const Ray& ray )
{
    return ray.o + hit.t * ray.d;
}


struct Triangle
{
    Point p;
    Vector e1, e2;
    int id;

    Triangle( const Point& _a, const Point& _b, const Point& _c, const int _id ) : p(_a), e1(Vector(_a, _b)), e2(Vector(_a, _c)), id(_id) {}

    /* calcule l'intersection ray/triangle
        cf "fast, minimum storage ray-triangle intersection"
        http://www.graphics.cornell.edu/pubs/1997/MT97.pdf

        renvoie faux s'il n'y a pas d'intersection valide (une intersection peut exister mais peut ne pas se trouver dans l'intervalle [0 htmax] du rayon.)
        renvoie vrai + les coordonnees barycentriques (u, v) du point d'intersection + sa position le long du rayon (t).
        convention barycentrique : p(u, v)= (1 - u - v) * a + u * b + v * c
    */
    Hit intersect( const Ray &ray, const float htmax ) const
    {
        Vector pvec= cross(ray.d, e2);
        float det= dot(e1, pvec);

        float inv_det= 1 / det;
        Vector tvec(p, ray.o);

        float u= dot(tvec, pvec) * inv_det;
        if(u < 0 || u > 1) return Hit();

        Vector qvec= cross(tvec, e1);
        float v= dot(ray.d, qvec) * inv_det;
        if(v < 0 || u + v > 1) return Hit();

        float t= dot(e2, qvec) * inv_det;
        if(t > htmax || t < 0) return Hit();

        return Hit(id, t, u, v);           // p(u, v)= (1 - u - v) * a + u * b + v * c
    }
};

// ensemble de triangles.
// a remplacer par un vrai bvh, cf tuto_bvh_simple et les codes plus structures : tuto_bvh, bvh, centroid_builder etc.
struct BVH
{
    std::vector<Triangle> triangles;

    BVH( ) = default;
    BVH( const Mesh& mesh ) { build(mesh); }

    void build( const Mesh& mesh )
    {
        triangles.clear();
        triangles.reserve(mesh.triangle_count());
        for(int id= 0; id < mesh.triangle_count(); id++)
        {
            TriangleData data= mesh.triangle(id);
            triangles.push_back( Triangle(data.a, data.b, data.c, id) );
        }
    }

    Hit intersect( const Ray& ray ) const
    {
        Hit hit;
        float tmax= ray.tmax;
        for(int id= 0; id < int(triangles.size()); id++)
            // ne renvoie vrai que si l'intersection existe dans l'intervalle [0 tmax]
            if(Hit h= triangles[id].intersect(ray, tmax))
            {
                hit= h;
                tmax= h.t;
            }

        return hit;
    }

    bool visible( const Ray& ray ) const
    {
        for(int id= 0; id < int(triangles.size()); id++)
            if(triangles[id].intersect(ray, ray.tmax))
                return false;

        return true;
    }
};


// construit un repere ortho tbn, a partir d'un seul vecteur, la normale / axe z...
// cf "generating a consistently oriented tangent space"
// http://people.compute.dtu.dk/jerf/papers/abstracts/onb.html
// cf "Building an Orthonormal Basis, Revisited", Pixar, 2017
// http://jcgt.org/published/0006/01/01/
struct World
{
    World( const Vector& _n ) : n(_n)
    {
        float sign= std::copysign(1.0f, n.z);
        float a= -1.0f / (sign + n.z);
        float d= n.x * n.y * a;
        t= Vector(1.0f + sign * n.x * n.x * a, sign * d, -sign * n.x);
        b= Vector(d, sign + n.y * n.y * a, -n.y);
    }

    // transforme le vecteur du repere local vers le repere du monde
    Vector operator( ) ( const Vector& local )  const { return local.x * t + local.y * b + local.z * n; }

    // transforme le vecteur du repere du monde vers le repere local
    Vector inverse( const Vector& global ) const { return Vector(dot(global, t), dot(global, b), dot(global, n)); }

    Vector t;
    Vector b;
    Vector n;
};


// genere une direction uniforme 1 / 2pi, cf GI compendium, eq 34
struct UniformDirection
{
    UniformDirection( const int _n, const Vector& _z ) : world(_z), n(_n){}

    Vector operator() ( const float u1, const float u2 ) const
    {
        float cos_theta= u1;
        float phi= 2.f * float(M_PI) * u2;
        float sin_theta= std::sqrt(std::max(0.f, 1.f - cos_theta*cos_theta));

        return world(Vector(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta));
    }

    float pdf( const Vector& v ) const { if(dot(v, world.n) < 0) return 0; else return 1.f / (2.f * float(M_PI)); }
    int size( ) const { return n; }

protected:
    World world;
    int n;
};


int main( const int argc, const char **argv )
{
    const char *mesh_filename= "cornell.obj";
    const char *orbiter_filename= "orbiter.txt";

    if(argc > 1) mesh_filename= argv[1];
    if(argc > 2) orbiter_filename= argv[2];

    printf("%s: '%s' '%s'\n", argv[0], mesh_filename, orbiter_filename);

    // creer l'image resultat
    Image image(1024, 640);

    // charger un objet
    Mesh mesh= read_mesh(mesh_filename);
    if(mesh.triangle_count() == 0)
        // erreur de chargement, pas de triangles
        return 1;

    // construire le bvh ou recuperer l'ensemble de triangles du mesh...
    BVH bvh(mesh);

    // charger la camera
    Orbiter camera;
    if(camera.read_orbiter(orbiter_filename))
        // erreur, pas de camera
        return 1;

    // recupere les transformations view, projection et viewport pour generer les rayons
    Transform m= Identity();
    Transform v= camera.view();
    Transform p= camera.projection(image.width(), image.height(), 45);

    Transform mvp  = p * v * m ;
    Transform mvpInv = mvp.inverse();

    auto cpu_start= std::chrono::high_resolution_clock::now();

    // parcourir tous les pixels de l'image
    // en parallele avec openMP, un thread par bloc de 16 lignes
#pragma omp parallel for schedule(dynamic, 16)
    for(int py= 0; py < image.height(); py++)
    {
        // nombres aleatoires, version c++11
        std::random_device seed;
        // un generateur par thread... pas de synchronisation
        std::mt19937 rng(seed());
        std::uniform_real_distribution<float> u01(0.f, 1.f);

        for(int px= 0; px < image.width(); px++)
        {
            // generer le rayon pour le pixel (x, y)
            float x= px + .5f;          // centre du pixel
            float y= py + .5f;

            Point o = camera.position();  // origine
            Point e = mvpInv( Point(x/512.0 - 1, y/320.0 - 1 , 1) ) ;  // extremite

            // calculer les intersections
            Ray ray(o, e);
            if(Hit hit= bvh.intersect(ray))
            {
                // recupere les donnees sur l'intersection
                TriangleData triangle= mesh.triangle(hit.triangle_id);
                Point p= point(hit, ray);               // point d'intersection
                Vector pn= normal(hit, triangle);       // normale interpolee du triangle au point d'intersection
                if(dot(pn, ray.d) > 0)                  // retourne la normale vers l'origine du rayon
                    pn= -pn;

                // couleur du pixel
                Color color= mesh.triangle_material(hit.triangle_id).diffuse * std::max(0.0f, dot(-pn, normalize(ray.d)) );

                // genere des directions autour de p
                UniformDirection directions(256, pn);            // genere des directions uniformes 1 / 2pi
                const float scale= 10;
                int N = directions.size();

                float sum = 0;
                for(int i= 0; i < directions.size(); i++)
                {
                    // genere une direction
                    Vector w= directions(u01(rng), u01(rng));

                    // teste le rayon dans cette direction
                    Ray shadow(p + pn * .001f, p + w * scale);

                    if(bvh.visible(shadow))
                    {
                        // calculer l'eclairage ambient :
                        // en partant de la formulation de l'eclairage direct :
                        // L_r(p, o) = \int L_i(p, w) * f_r(p, w -> o) * cos \theta dw
                        // avec
                        // L_i(p, w) = L_e(hit(p, w), -w) * V(hit(p, w), p)
                        // L_e(q, v) = 1
                        // donc L_i(p, w)= V(hit(p, w), p)
                        // f_r() = 1 / pi

                        // donc
                        // L_r(p, w)= \int 1/pi * V(hit(p, w), p) cos \theta dw

                        // et l'estimateur monte carlo s'ecrit :
                        // L_r(p, w)= 1/n \sum { 1/pi * V(hit(p, w_i), p) * cos \theta_i } / { pdf(w_i) }
                        float cosTheta = 1.0f - (2.0f*i + 1.0f)/(2.0f*N);
                        sum += (1.0f/3.1419)*cosTheta/ directions.pdf(w);
                    }
                }

                color = color*Color(sum/N) ;

                image(px, py)= Color(color, 1);
            }
        }
    }

    auto cpu_stop= std::chrono::high_resolution_clock::now();
    int cpu_time= std::chrono::duration_cast<std::chrono::milliseconds>(cpu_stop - cpu_start).count();
    printf("cpu  %ds %03dms\n", int(cpu_time / 1000), int(cpu_time % 1000));

    // enregistrer l'image resultat
    write_image(image, "render.png");
    write_image_hdr(image, "render.hdr");

    return 0;
}
