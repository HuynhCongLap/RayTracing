
#ifndef _RAY_H
#define _RAY_H

#include <cfloat>

#include "vec.h"


struct Ray
{
    Point o;
    Vector d;
    float tmax;

    Ray( const Point& _o, const Point& _e ) : o(_o), d(Vector(_o, _e)), tmax(1) {}
    Ray( const Point& _o, const Vector& _d ) : o(_o), d(_d), tmax(FLT_MAX) {}
};

struct Hit
{
    int triangle_id;
    float t;
    float u, v;

    Hit( ) : triangle_id(-1), t(0), u(0), v(0) {}       // pas d'intersection
    Hit( const int _id, const float _t, const float _u, const float _v ) : triangle_id(_id), t(_t), u(_u), v(_v) {}

    operator bool( ) const { return (triangle_id != -1); }      // renvoie vrai si l'intersection est initialisee...
};

#endif
