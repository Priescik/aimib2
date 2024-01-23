glcompat_attrs=4

[VERTEX]
#include "common.glsl"

attribute mediump vec4 a_vert;
attribute vec2 a_texcoord;
varying mediump vec2 v_texcoord;
varying mediump vec2 v_texcoord2;
varying mediump vec2 v_texcoord3;
varying mediump vec2 v_texcoord4;

void main()
{
gl_Position = u_projmodmat * a_vert;
v_texcoord=a_texcoord;
v_texcoord4=v_texcoord+vec2(0.0,-u_offset);
v_texcoord3=v_texcoord+vec2(0.0,-u_offset*0.33);
v_texcoord2=v_texcoord+vec2(0.0,u_offset*0.33);
v_texcoord=v_texcoord+vec2(0.0,u_offset);
}
		
[FRAGMENT]
#include "common.glsl"

varying mediump vec2 v_texcoord;
varying mediump vec2 v_texcoord2;
varying mediump vec2 v_texcoord3;
varying mediump vec2 v_texcoord4;

void main()
{
lowp float x=texture2D(u_tex,v_texcoord).r*0.15+texture2D(u_tex,v_texcoord2).r*0.35+texture2D(u_tex,v_texcoord3).r*0.35+texture2D(u_tex,v_texcoord4).r*0.15;
gl_FragColor=vec4(x,x,x,1.0);
}
