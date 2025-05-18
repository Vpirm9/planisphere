#!/usr/bin/python3
# holder.py
# -*- coding: utf-8 -*-
#
# The python script in this file makes the various parts of a model planisphere.
#
# Copyright (C) 2014-2024 Dominic Ford <https://dcford.org.uk/>
#
# This code is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# You should have received a copy of the GNU General Public License along with
# this file; if not, write to the Free Software Foundation, Inc., 51 Franklin
# Street, Fifth Floor, Boston, MA  02110-1301, USA

# ----------------------------------------------------------------------------

"""
Render the holder for the planisphere.
"""

from math import pi, sin, cos, atan2, asin, hypot, sqrt
from numpy import arange, concatenate, sum, array, arctan2, unwrap, degrees, argsort #,sqrt
from typing import Dict, List, Tuple

from constants import radius, transform, pos
from constants import unit_deg, unit_rev, unit_cm, unit_mm, r_1, r_2, fold_gap, central_hole_size, line_width_base, font_size_base
from graphics_context import BaseComponent, GraphicsContext
from settings import fetch_command_line_arguments
from text import text

fold_gap=0.0

class Holder(BaseComponent):
    """
    Render the holder for the planisphere.
    """

    def default_filename(self) -> str:
        """
        Return the default filename to use when saving this component.
        """
        return "holder"

    def bounding_box(self, settings: dict) -> Dict[str, float]:
        """
        Return the bounding box of the canvas area used by this component.

        :param settings:
            A dictionary of settings required by the renderer.
        :return:
         Dictionary with the elements 'x_min', 'x_max', 'y_min' and 'y_max' set
        """

        h: float = r_1 + fold_gap

        #return {
            #'x_min': -r_1 - 4 * unit_mm,
            #'x_max': r_1 + 4 * unit_mm,
            #'y_min': -r_2 - h - 4 * unit_mm,
            #'y_max': h + 1.2 * unit_cm
        return {
            'x_min': -r_1 - 4 * unit_mm,
            'x_max': r_1 + 4 * unit_mm,
            'y_min': -r_1 - 4 * unit_mm - h, #-r_2 - h - 4 * unit_mm - 4* unit_cm,
            'y_max': r_1 + 4 * unit_mm - h #h + 1.2 * unit_cm - 4* unit_cm
        }
        #}

    def draw_alt_az_grid(self, context: GraphicsContext, latitude: float, x0: Tuple[float, float]) -> None:
            """Draw the alt-az grid directly into the viewing window."""
            # Set grid parameters based on latitude
            alt_edge = -12 if abs(latitude) >= 15 else -9
            azimuth_step = 1

            # Draw grid lines (semi-transparent and thin)
            #context.set_color((0.3, 0.3, 0.3, 0.5))  # Light gray, 50% opacity ---> Changed to black
            context.set_color((0, 0, 0, 1))  # Solid black
            context.set_line_width(0.25 * line_width_base)


            # Altitude circles (20° increments)
            for alt in range(10, 85, 20):
                path = [
                    transform(alt=alt, az=az, latitude=latitude)
                    for az in arange(0, 360.5, 1)
                ]
                context.begin_path()
                for i, p in enumerate(path):
                    r = radius(dec=p[1] / unit_deg, latitude=latitude)
                    pos_abs = pos(r, p[0])
                    x = x0[0] + pos_abs['x']
                    y = -x0[1] + pos_abs['y']
                    if i == 0:
                        context.move_to(x, y)
                    else:
                        context.line_to(x, y)
                context.stroke()

            # Azimuth lines (30° increments) - dashed
            context.set_line_style(dotted=True)
            for az in arange(0, 359, 30):
                path = [
                    transform(alt=alt, az=az, latitude=latitude)
                    for alt in arange(0, 90.1, 1)
                ]
                context.begin_path()
                for i, p in enumerate(path):
                    r = radius(dec=p[1] / unit_deg, latitude=latitude)
                    pos_abs = pos(r, p[0])
                    x = x0[0] + pos_abs['x']
                    y = -x0[1] + pos_abs['y']
                    if i == 0:
                        context.move_to(x, y)
                    else:
                        context.line_to(x, y)
                context.stroke()
            context.set_line_style(dotted=False)  # Reset line style
            context.set_color((0.3, 0.3, 0.3, 0.5))



    def do_rendering(self, settings: dict, context: GraphicsContext) -> None:
        """
        This method is required to actually render this item.

        :param settings:
            A dictionary of settings required by the renderer.
        :param context:
            A GraphicsContext object to use for drawing
        :return:
            None
        """

        is_southern: bool = settings['latitude'] < 0
        latitude: float = abs(settings['latitude'])
        language: str = settings['language']







        context.set_font_size(0.9)

        a: float = 6 * unit_cm
        h: float = r_1 + fold_gap

        # Added for grid: 
        x0 = (0, h)

        context.begin_path()
        context.arc(centre_x=0, centre_y=-h, radius=r_1,  # Positioned at viewing window center
                    arc_from=0, arc_to=2*pi)
        context.set_color((0, 0, 0, 1))
        context.stroke(line_width=1.5 * line_width_base)

        # Draw dotted line for folding the bottom of the planisphere
        #context.begin_path()
        #context.move_to(x=-r_1, y=0)
        #context.line_to(x=r_1, y=0)
        #context.stroke(dotted=True)

        # Draw the rectangular back and lower body of the planisphere
        #context.begin_path()
        #context.move_to(x=-r_1, y=a)
        #context.line_to(x=-r_1, y=-a)
        #context.move_to(x=r_1, y=a)
        #context.line_to(x=r_1, y=-a)
        context.stroke(dotted=False)

        # Draw the curved upper part of the body of the planisphere
        theta: float = pi#unit_rev / 2 - atan2(r_1, h - a)
        context.begin_path()
        context.arc(centre_x=0, centre_y=-h, radius=r_2, arc_from=-theta - pi / 2, arc_to=theta - pi / 2)
        context.move_to(x=-r_2 * sin(theta), y=-h - r_2 * cos(theta))
        #context.line_to(x=-r_1, y=-a)
        #context.move_to(x=r_2 * sin(theta), y=-h - r_2 * cos(theta))
        #context.line_to(x=r_1, y=-a)
        #context.stroke()
        self.draw_alt_az_grid(context, latitude, x0)
        # Shade the viewing window which needs to be cut out
        #x0: Tuple[float, float] = (0, h)
        #context.begin_path()
        #i: int
        #az: float
        #for i, az in enumerate(arange(0, 360.5, 1)):
        #    pp: Tuple[float, float] = transform(alt=0, az=az, latitude=latitude)
        #    r: float = radius(dec=pp[1] / unit_deg, latitude=latitude)
        #    p: Dict[str, float] = pos(r=r, t=pp[0])
        #    if i == 0:
        #        context.move_to(x0[0] + p['x'], -x0[1] + p['y'])
        #    else:
        #        context.line_to(x0[0] + p['x'], -x0[1] + p['y'])
        #context.stroke()
        #context.fill(color=(0, 0, 0, 0.2))

        # Define the hour circle radius (r_2 is the outer edge of the dashed scale)
        hour_circle_radius = r_2 - 4 * unit_mm - font_size_base



        gray_color = (0, 0, 0, 0.2)  # Semi-transparent gray

        # First draw the viewing window edge
        x0: Tuple[float, float] = (0, h)
        
        # Create the viewing window path
        viewing_window_points = []
        for az in arange(0, 360, 1):
            pp = transform(alt=0, az=az, latitude=latitude)
            r = radius(dec=pp[1] / unit_deg, latitude=latitude)
            p = pos(r=r, t=pp[0])
            viewing_window_points.append((x0[0] + p['x'], -x0[1] + p['y']))

        # Create the hour circle path
        hour_circle_points = []
        for angle in arange(theta - pi/2, -theta - pi/2, -0.01):
            x = hour_circle_radius * cos(angle)
            y = -h + hour_circle_radius * sin(angle)
            hour_circle_points.append((x, y))

        # Draw the filled area between the two paths (without stroke)
        context.begin_path()
        context.move_to(*viewing_window_points[0])
        for x, y in viewing_window_points:
            context.line_to(x, y)
        context.close_path()
        context.line_to(*viewing_window_points[0])
        for x, y in hour_circle_points:
            context.line_to(x, y)
        context.line_to(*hour_circle_points[0])
        context.close_path()
        context.fill(color=gray_color)

        # Make the last triangle
        #context.begin_path()
        #context.move_to(*viewing_window_points[0])
        #context.line_to(*hour_circle_points[0])
        #context.line_to(*hour_circle_points[-1])
        #context.close_path()
        #context.fill(color=gray_color)

        # Now draw the edges with proper colors
        # Viewing window edge in gray

        context.begin_path()
        context.move_to(*viewing_window_points[0])
        for x, y in viewing_window_points[1:]:
            context.line_to(x, y)
        context.line_to(*viewing_window_points[0])
        context.set_color((0, 0, 0, 1))#context.set_color(gray_color)
        context.set_line_width(line_width_base)
        context.stroke()


        # Start and end of astronomical night
        alt_twilight = -18  # Altitude for line below horizon
        twilight_points = []

        
        # From 340 to 360 (inclusive)
        part1 = arange(90, 0, -1)
        # From 0 to 200 (inclusive)
        part2 = arange(360, 270, -1)
        # Concatenate both parts
        az_points_right = concatenate((part1, part2))
        az_points_left = arange(90, 270, 1)


        # Altitude circles (20° increments)
        alt=-18.0
        last_i_shown=0

        path = [
            transform(alt=alt, az=az, latitude=latitude)
            for az in az_points_left
        ]
        context.begin_path()
        for i, p in enumerate(path):
            r = radius(dec=p[1] / unit_deg, latitude=latitude)
            pos_abs = pos(r, p[0])
            x = x0[0] + pos_abs['x']
            y = -x0[1] + pos_abs['y']
            if i == 0:
                context.move_to(x, y)
            else:
                if sqrt((pos_abs['x'])**2+(pos_abs['y'])**2) <= (r_2-10.0*unit_mm):
                    context.line_to(x, y)
                    last_i_shown=i
        context.stroke()

        twilight_text_pos_index=2*(last_i_shown//3)



        # Compute rotation and position for twilight label
        i = twilight_text_pos_index
        az1, az2 = az_points_left[i - 1], az_points_left[i]
        p1, p2 = [transform(alt=alt, az=az, latitude=latitude) for az in (az1, az2)]
        r1, r2 = [radius(dec=p[1] / unit_deg, latitude=latitude) for p in (p1, p2)]
        pos1, pos2 = [pos(r, p[0]) for r, p in zip((r1, r2), (p1, p2))]

        dx, dy = pos2['x'] - pos1['x'], pos2['y'] - pos1['y']
        tr = -unit_rev / 4 - atan2(dx, dy)

        context.set_font_size(0.6)
        context.set_font_style(bold=True)
        context.set_color((0, 0, 0, 1))
        context.text(text="End of astronomical night",
                     x=x0[0] + pos2['x'], y=-x0[1] + pos2['y'],
                     h_align=0, v_align=1,
                     gap=(+4 * unit_mm - font_size_base),
                     rotation=tr)
        context.set_font_style(bold=False)
        context.set_font_size(0.9)


        # Right side
        path = [
            transform(alt=alt, az=az, latitude=latitude)
            for az in az_points_right
        ]
        context.begin_path()
        for i, p in enumerate(path):
            r = radius(dec=p[1] / unit_deg, latitude=latitude)
            pos_abs = pos(r, p[0])
            x = x0[0] + pos_abs['x']
            y = -x0[1] + pos_abs['y']
            if i == 0:
                context.move_to(x, y)
            else:
                if sqrt((pos_abs['x'])**2+(pos_abs['y'])**2) <= (r_2-10.0*unit_mm):
                    context.line_to(x, y)
        context.stroke()





        #for az in arange(90, 270, 1):
        #    pp = transform(alt=alt_twilight, az=az, latitude=latitude)
        #    r = radius(dec=pp[1] / unit_deg, latitude=latitude)
        #    p = pos(r=r, t=pp[0])
        #    twilight_points.append((x0[0] + p['x'], -x0[1] + p['y']))
    #
        #context.begin_path()
        #context.move_to(*twilight_points[0])    
        #for x, y in twilight_points:
        #    if sqrt(x**2+(y)**2) <= (r_1):
        #        context.line_to(x, y)
        #context.set_color((0., 0., 0., 1.))  
        #context.set_line_width(1.5 * line_width_base)
        #context.stroke()

        
        #twilight_points_array=array(twilight_points)
        #context.begin_path()
        #context.move_to(*filtered_twilight_points[0])

        #r_squared = sum(twilight_points_array**2, axis=1)
        #r_squared = twilight_points_array[:, 0]**2 + (twilight_points_array[:, 1] - h)**2
        

        # Keep only points where r <= r_border
        #filtered_twilight_points = twilight_points_array[r_squared <= r_2**2]





        #context.begin_path()
        #context.move_to(*filtered_twilight_points[0])
        #for x, y in filtered_twilight_points[1:]:
        #    context.line_to(x, y)
        #context.set_color((0., 0., 0., 1.))  # Dark blue line
        #context.set_line_width(1.5 * line_width_base)
        #context.stroke()



        ## Hour circle edge in black ---> Might be needed
        #context.begin_path()
        #context.move_to(*hour_circle_points[0])
        #for x, y in hour_circle_points[1:]:
        #    context.line_to(x, y)
        #context.set_color((0, 0, 0, 1))
        #context.stroke()


        # Display instructions for cutting out the viewing window
        #instructions: str = text[language]["cut_out_instructions"]
        #context.set_color(color=(0, 0, 0, 1))
        #context.text_wrapped(text=instructions,
        #                     width=4 * unit_cm, justify=0,
        #                     x=0, y=-h - r_1 * 0.35,
        #                     h_align=0, v_align=0, rotation=0)

        # Cardinal points

        def cardinal(dir: str, ang: float) -> None:
            
            scale_factor=1.0

            #pp: Tuple[float, float] = transform(alt=0, az=ang - 0.01, latitude=latitude)
            #r: float = radius(dec=pp[1] / unit_deg, latitude=latitude)
            #p: Dict[str, float] = pos(r, pp[0])

            #pp2: Tuple[float, float] = transform(alt=0, az=ang + 0.01, latitude=latitude)
            #r2: float = radius(dec=pp2[1] / unit_deg, latitude=latitude)
            #p2: Dict[str, float] = pos(r, t=pp2[0])

            #p3: List[float] = [p2[i] - p[i] for i in ('x', 'y')]
            #tr: float = -unit_rev / 4 - atan2(p3[0], p3[1])
                # Calculate position for azimuth (ang - 0.01°)
            pp: Tuple[float, float] = transform(alt=0, az=ang - 0.01, latitude=latitude)
            r: float = radius(dec=pp[1] / unit_deg, latitude=latitude) * scale_factor  # Apply scale_factor
            p: Dict[str, float] = pos(r, pp[0])

            # Calculate position for azimuth (ang + 0.01°)
            pp2: Tuple[float, float] = transform(alt=0, az=ang + 0.01, latitude=latitude)
            r2: float = radius(dec=pp2[1] / unit_deg, latitude=latitude) * scale_factor  # Apply scale_factor
            p2: Dict[str, float] = pos(r2, t=pp2[0])

            # Compute tangent vector and text rotation
            p3: List[float] = [p2[i] - p[i] for i in ('x', 'y')]
            tr: float = -unit_rev / 4 - atan2(p3[0], p3[1])

            context.set_color((0, 0, 0, 1))
            context.text(text=dir, x=x0[0] + p['x'], y=-x0[1] + p['y'],
                         h_align=0, v_align=1, gap=(-1*unit_mm-font_size_base), rotation=tr)



        # Write the cardinal points around the horizon of the viewing window
        context.set_font_style(bold=True)

        txt: str = text[language]['cardinal_points']['w']
        cardinal(txt, 180 if not is_southern else 0)

        txt = text[language]['cardinal_points']['s']
        cardinal(txt, 270 if not is_southern else 90)

        txt = text[language]['cardinal_points']['e']
        cardinal(txt, 0 if not is_southern else 180)

        txt = text[language]['cardinal_points']['n']
        cardinal(txt, 90 if not is_southern else 270)

        context.set_font_style(bold=False)

        # Clock face, which lines up with the date scale on the star wheel
        theta: float = unit_rev / 24 * 12 # Modified original write for whole day Orignal#7  # 5pm -> 7am means we cover 7 hours on either side of midnight
        dash: float = unit_rev / 24 / 12 # Modifed made for 5 min #4  Original: # Draw fat dashes at 15 minute intervals

        # Outer edge of dashed scale
        r_3: float = r_2 - 2 * unit_mm

        # Inner edge of dashed scale
        r_4: float = r_2 - 3 * unit_mm

        # Radius of dashes for marking hours
        r_5: float = r_2 - 4 * unit_mm

        # Radius of text marking hours
        r_6: float = r_2 - 5.5 * unit_mm

        # Inner and outer curves around dashed scale
        context.begin_path()
        context.arc(centre_x=0, centre_y=-h, radius=r_3, arc_from=-theta - pi / 2, arc_to=theta - pi / 2)
        context.begin_sub_path()
        context.arc(centre_x=0, centre_y=-h, radius=r_4, arc_from=-theta - pi / 2, arc_to=theta - pi / 2)
        context.stroke()

        # Draw a fat dashed line with one dash every 15 minutes
        for i in arange(-theta, theta, 2 * dash):
            context.begin_path()
            context.arc(centre_x=0, centre_y=-h, radius=(r_3 + r_4) / 2, arc_from=i - pi / 2, arc_to=i + dash - pi / 2)
            context.stroke(line_width=(r_3 - r_4) / line_width_base)

        # Write the hours
        #for hr in arange(-7, 7.1, 1):
        #    txt: str = "{:.0f}{}".format(hr if (hr > 0) else hr + 12,
        #                                 "AM" if (hr > 0) else "PM")
        #    if language == "fr":
        #        txt = "{:02d}h00".format(int(hr if (hr > 0) else hr + 24))
        #    if hr == 0:
        #        txt = ""
        #    t: float = unit_rev / 24 * hr * (-1 if not is_southern else 1)
#
        #    # Stroke a dash and write the number of the hour
        #    context.begin_path()
        #    context.move_to(x=r_3 * sin(t), y=-h - r_3 * cos(t))
        #    context.line_to(x=r_5 * sin(t), y=-h - r_5 * cos(t))
        #    context.stroke(line_width=1)
        #    context.text(text=txt, x=r_6 * sin(t), y=-h - r_6 * cos(t), h_align=0, v_align=0, gap=0, rotation=t)
        # Write the hours in 24-hour format


        minute_intervals = [0.25, 0.5, 0.75]  # 15, 30, 45 minutes as fractions of an hour
        minute_labels = ["15'", "30'", "45'"]    # Corresponding labels


        #minute_intervals = [0.3333, 0.6666,]  # 15, 30, 45 minutes as fractions of an hour
        #minute_labels = ["20'", "40'"]    # Corresponding labels

        r_7: float = r_2 - 3.5 * unit_mm  # Radius for minute ticks
        r_8: float = r_2 - 4.5 * unit_mm  # Radius for minute labels

        #original_font_size = context.get_font_size()
        #Add minutes
        
        context.set_font_size(0.6)

        for hr in range(24):
            for i, minute_frac in enumerate(minute_intervals):
                # Calculate position for this minute mark
                t: float = unit_rev / 24 * (hr + minute_frac) * (-1 if not is_southern else 1)

                # Draw minute tick

                context.begin_path()
                context.move_to(x=r_3 * sin(t), y=-h - r_3 * cos(t))
                context.line_to(x=r_7 * sin(t), y=-h - r_7 * cos(t))
                context.stroke(line_width=1)

                # Add minute label
                
                context.text(
                    text=minute_labels[i],
                    x=r_8 * sin(t), y=-h - r_8 * cos(t),
                    h_align=0, v_align=0, gap=0, rotation=t
                )
        # Add hours

        context.set_font_size(0.9)

        for hr in range(24):
                   
            #for i, minute_frac in enumerate(minute_intervals):
            #    # Calculate position for this minute mark
            #    t_min: float = unit_rev / 24 * ( minute_frac) * (-1 if not is_southern else 1)
            #    
            #    # Draw thicker tick for minute marks
            #    context.begin_path()
            #    context.move_to(x=r_3 * sin(t_min), y=-h - r_3 * cos(t_min))
            #    context.line_to(x=r_5 * sin(t_min), y=-h - r_4 * cos(t_min))
            #    context.stroke(line_width=1)  # Thicker line for minute marks
            #    
            #    # Add minute label
            #    context.set_font_size(0.5 * font_size_base)  # Smaller font for minute labels
            #    context.text(
            #        text=minute_labels[i],
            #        x=r_6 * sin(t_min), y=-h - r_6 * cos(t_min),
            #        h_align=0, v_align=0, gap=0, rotation=t_min
            #    )

            ## Write the hours in 24-hour format
            #context.set_font_size(font_size_base)  # Reset to normal font size



            #txt: str = "{:02d}h".format(hr)
            txt: str = "{}h".format(hr)  # e.g., 8h
            if language == "fr":
                txt = "{:02d}h00".format(hr)

            t: float = unit_rev / 24 * hr * (-1 if not is_southern else 1)

            # Stroke a dash and write the number of the hour
            context.begin_path()
            context.move_to(x=r_3 * sin(t), y=-h - r_3 * cos(t))
            context.line_to(x=r_5 * sin(t), y=-h - r_5 * cos(t))
            context.stroke(line_width=1)

            if hr != 24:
                context.text(
                    text=txt,
                    x=r_6 * sin(t), y=-h - r_6 * cos(t),
                    h_align=0, v_align=0, gap=0, rotation=t
                )
        # Back edge
        #b: float = unit_cm
        #t1: float = atan2(h - a, r_1)
        #t2: float = asin(b / hypot(r_1, h - a))
        #context.begin_path()
        #context.move_to(x=-r_1, y=a)
        #context.line_to(x=-b * sin(t1 + t2), y=h + b * cos(t1 + t2))
        #context.move_to(x=r_1, y=a)
        #context.line_to(x=b * sin(t1 + t2), y=h + b * cos(t1 + t2))
        #context.arc(centre_x=0, centre_y=h, radius=b, arc_from=unit_rev / 2 - (t1 + t2) - pi / 2,
        #            arc_to=unit_rev / 2 + (t1 + t2) - pi / 2)
        #context.stroke(line_width=1)

        # For latitudes not too close to the pole, we have enough space to fit instructions onto the planisphere
        #if latitude < 56:
        #    # Big bold title
        #    context.set_font_size(3.0)
        #    txt: str = text[language]['title']
        #    context.set_font_style(bold=True)
        #    context.text(
        #        text="{} {:.0f}\u00B0{}".format(txt, float(latitude), "N" if not is_southern else "S"),
        #        x=0, y=-4.8 * unit_cm,
        #        h_align=0, v_align=0, gap=0, rotation=0)
        #    context.set_font_style(bold=False)
#
        #    # First column of instructions
        #    context.set_font_size(2)
        #    context.text(
        #        text="1",
        #        x=-5.0 * unit_cm, y=-4.0 * unit_cm,
        #        h_align=0, v_align=0, gap=0, rotation=0)
        #    context.set_font_size(1)
        #    context.text_wrapped(
        #        text=text[language]['instructions_1'],
        #        x=-5.0 * unit_cm, y=-3.4 * unit_cm, width=4.5 * unit_cm, justify=-1,
        #        h_align=0, v_align=1, rotation=0)
#
        #    # Second column of instructions
        #    context.set_font_size(2)
        #    context.text(
        #        text="2",
        #        x=0, y=-4.0 * unit_cm,
        #        h_align=0, v_align=0, gap=0, rotation=0)
        #    context.set_font_size(1)
        #    context.text_wrapped(
        #        text=text[language]['instructions_2'].format(cardinal="north" if not is_southern else "south"),
        #        x=0, y=-3.4 * unit_cm, width=4.5 * unit_cm, justify=-1,
        #        h_align=0, v_align=1, rotation=0)
#
        #    # Third column of instructions
        #    context.set_font_size(2)
        #    context.text(
        #        text="3",
        #        x=5.0 * unit_cm, y=-4.0 * unit_cm,
        #        h_align=0, v_align=0, gap=0, rotation=0)
        #    context.set_font_size(1)
        #    context.text_wrapped(
        #        text=text[language]['instructions_3'],
        #        x=5.0 * unit_cm, y=-3.4 * unit_cm, width=4.5 * unit_cm, justify=-1,
        #        h_align=0, v_align=1, rotation=0)
        #else:
        #    # For planispheres for use at high latitudes, we don't have much space, so don't show instructions.
        #    # We just display a big bold title
        #    context.set_font_size(3.0)
        #    txt = text[language]['title']
        #    context.set_font_style(bold=True)
        #    context.text(
        #        text="%s %d\u00B0%s" % (txt, latitude, "N" if not is_southern else "S"),
        #        x=0, y=-1.8 * unit_cm,
        #        h_align=0, v_align=0, gap=0, rotation=0)
        #    context.set_font_style(bold=False)

        # Write explanatory text on the back of the planisphere
        #context.set_font_size(1.1)
        #context.text_wrapped(
        #    text=text[language]['instructions_4'],
        #    x=0, y=5.5 * unit_cm, width=12 * unit_cm, justify=-1,
        #    h_align=0, v_align=1, rotation=0.5 * unit_rev)

        # Display web link and copyright text
        #txt = text[language]['more_info']
        #context.set_font_size(0.9)
        #context.text(text=txt, x=0, y=-0.5 * unit_cm, h_align=0, v_align=0, gap=0, rotation=0)
        #context.set_font_size(0.9)
        #context.text(text=txt, x=0, y=0.5 * unit_cm, h_align=0, v_align=0, gap=0, rotation=pi)

        # Draw central hole
        #context.begin_path()
        #context.circle(centre_x=0, centre_y=h, radius=central_hole_size)
        #context.stroke()


# Do it right away if we're run as a script
if __name__ == "__main__":
    # Fetch command line arguments passed to us
    arguments = fetch_command_line_arguments(default_filename=Holder().default_filename())

    # Render the holder for the planisphere
    Holder(settings={
        'latitude': arguments['latitude'],
        'language': 'en'
    }).render_to_file(
        filename=arguments['filename'],
        img_format=arguments['img_format']
    )
