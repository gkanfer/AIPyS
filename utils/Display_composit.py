import numpy as np
from skimage.exposure import rescale_intensity, histogram
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.express as px
from skimage import io, filters, measure, color, img_as_ubyte

from utils import display_and_xml as dx


def image_with_contour(img, active_labels, data_table, active_columns, color_column):
    """
    Returns a greyscale image that is segmented and superimposed with contour traces of
    the segmented regions, color coded by values from a data table.
    Parameters
    ----------
    img : PIL Image object.
    active_labels : list
        the currently visible labels in the datatable
    data_table : pandas.DataFrame
        the currently visible entries of the datatable
    active_columns: list
        the currently selected columns of the datatable
    color_column: str
        name of the datatable column that is used to define the colorscale of the overlay
    """

    # First we get the values from the selected datatable column and use them to define a colormap
    values = np.array(data_table[color_column].values)
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.cm.get_cmap("plasma")

    # Now we convert our background image to a greyscale bytestring that is very small and can be transferred very
    # efficiently over the network. We do not want any hover-information for this image, so we disable it
    fig = px.imshow(img, binary_string=True, binary_backend="jpg",)
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    # For each region that is visible in the datatable, we compute and draw the filled contour, color it based on
    # the color_column value of this region, and add it to the figure
    # here is an small tutorial of this: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
    for rid, row in data_table.iterrows():
        label = row.label
        value = row[color_column]
        contour = measure.find_contours(active_labels == label, 0.5)[0]
        # We need to move the contour left and up by one, because
        # we padded the label array
        y, x = contour.T - 1
        # We add the values of the selected datatable columns to the hover information of the current region
        hoverinfo = (
            "<br>".join(
                [
                    # All numbers are passed as floats. If there are no decimals, cast to int for visibility
                    f"{prop_name}: {f'{int(prop_val):d}' if prop_val.is_integer() else f'{prop_val:.3f}'}"
                    if np.issubdtype(type(prop_val), "float")
                    else f"{prop_name}: {prop_val}"
                    for prop_name, prop_val in row[active_columns].iteritems()
                ]
            )
            # remove the trace name. See e.g. https://plotly.com/python/reference/#scatter-hovertemplate
            + " <extra></extra>"
        )
        fig.add_scatter(
            x=x,
            y=y,
            name=label,
            opacity=0.2,
            mode="lines",
            line=dict(color=mpl.colors.rgb2hex(cmap(norm(value))),),
            fill="toself",
            customdata=[label] * len(x),
            showlegend=False,
            hovertemplate=hoverinfo,
            hoveron="points+fills",
        )

    # Finally, because we color our contour traces one by one, we need to manually add a colorscale to explain the
    # mapping of our color_column values to the colormap. This also gets added to the figure
    fig.add_scatter(
        # We only care about the colorscale here, so the x and y values can be empty
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            colorscale=[mpl.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 50)],
            showscale=True,
            # The cmin and cmax values here are arbitrary, we just set them to put our value ticks in the right place
            cmin=-5,
            cmax=5,
            colorbar=dict(
                tickvals=[-5, 5],
                ticktext=[f"{np.min(values[values!=0]):.2f}", f"{np.max(values):.2f}",],
                # We want our colorbar to scale with the image when it is resized, so we set them to
                # be a fraction of the total image container
                lenmode="fraction",
                len=0.6,
                thicknessmode="fraction",
                thickness=0.05,
                outlinewidth=1,
                # And finally we give the colorbar a title so the user may know what value the colormap is based on
                title=dict(text=f"<b>{color_column.capitalize()}</b>"),
            ),
        ),
        hoverinfo="none",
    )

    # Remove axis ticks and labels and have the image fill the container
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), template="simple_white")
    fig.update_xaxes(visible=False, range=[0, img.width]).update_yaxes(
        visible=False, range=[img.height, 0]
    )
    return fig
#https://dash.plotly.com/datatable/conditional-formatting
def row_highlight(roi_list_ctrl,roi_list_target):
    '''
        Componant of Dash datatable - highlight raws in the table
        :parameter
        roi_list_ctrl - list of ROI - in red #F31515
        roi_list_target -  list of ROI - in green #1ABA19
    '''
    return  ([
                 {'if': {'filter_query': '{{label}} = {}'.format(int(roi_ctrl))},
                     'backgroundColor': '{}'.format('#F31515'),
                     'color': 'white'
                 }
                for roi_ctrl in roi_list_ctrl
                ] +
                [
                 {
                     'if':
                         {'filter_query': '{{label}} = {}'.format(int(roi_))},
                        'backgroundColor': '{}'.format('#1ABA19'),
                        'color': 'white'
                 }
                for roi_ in roi_list_target
            ])

def countor_map(mask_target,roi_ctrl,roi_target,ch2_rgb):
    ''':parameter
        mask_target - contour target channel
        ROI - current click point and the list of the last clicks
        ch2_rgb - with seed is displayed in blue
        return:
        an rGB image with seed and clicked target segment map.
    '''
    if len(roi_ctrl) > 0:
        bf_mask_sel_ctrl = np.zeros(np.shape(mask_target), dtype=np.int32)
        #adding ctrl map
        for list in roi_ctrl:
            bf_mask_sel_ctrl[mask_target == list] = list
        c_mask_ctrl = dx.binary_frame_mask(ch2_rgb, bf_mask_sel_ctrl)
    else:
        c_mask_ctrl = np.zeros(np.shape(mask_target), dtype=np.int32)
    # adding target map
    if len(roi_target) > 0:
        bf_mask_sel_trgt = np.zeros(np.shape(mask_target), dtype=np.int32)
        for list in roi_target:
            bf_mask_sel_trgt[mask_target == list] = list
        c_mask_trgt = dx.binary_frame_mask(ch2_rgb, bf_mask_sel_trgt)
    else:
        c_mask_trgt = np.zeros(np.shape(mask_target), dtype=np.int32)
    ch2_rgb[c_mask_ctrl > 0, 0] = 255
    ch2_rgb[c_mask_trgt > 0, 1] = 255
    return ch2_rgb



