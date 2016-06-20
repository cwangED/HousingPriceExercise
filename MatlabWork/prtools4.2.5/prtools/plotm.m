%PLOTM Plot mapping values, contours or surface
% 
% 	H = PLOTM(W,S,N)
%
% INPUT
%   W   Trained mapping
%   S   Plot strings, or scalar selecting type of plot 
%          1: density plot;
%          2: contour plot (default); 
%          3: 3D surface plot; 
%          4: 3D surface plot above 2D contour plot; 
%          5; 3D mesh plot;
%          6: 3D mesh plot above 2D contour plot)
%   N   Contour level(s) to plot 
%         (default: 10 contours between minimum and maximum)
%
% OUTPUT
%		H		Array of graphics handles
%
% DESCRIPTION
% This routine, similar to PLOTC, plots contours (not just decision
% boundaries) of the mapping W on predefined axis, typically generated by
% SCATTERD. Plotstrings may be set in S. The vector N selects the contour.
% 
% EXAMPLES
% See PREX_DENSITY
%
% SEE ALSO
% MAPPINGS, SCATTERD, PLOTC

% Copyright: R.P.W. Duin, r.p.w.duin@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

% $Id: plotm.m,v 1.4 2009/09/25 13:15:17 duin Exp $

function handle = plotm(w,arg2,n,cnd)

	prtrace(mfilename);
      
	ismapping(w);				% Assert that W is a mapping.
  w = w*setbatch;     % Avoid memory prolems with large gridsizes

	% Get the parameters, the plotstrings and the number of contours.

    if (nargin < 4)
        cnd = 1;
    end;   
    
	[k,c] = size(w);
	if (nargin < 3)
		n = []; 
	end

	plottype = 2; s = []; 
	if (nargin >= 2)
		if (~isstr(arg2) & ~isempty(arg2))
			plottype = arg2; 
		else
			s = arg2;
		end
	end
	
	if plottype == 2 & size(w,1) == 1
		plottype = 1;
	end
	
	if (nargin < 2) | (isempty(s)) 
		col = 'brmk'; 
		s = [col' repmat('-',4,1)];
		s = char(s,[col' repmat('--',4,1)]);
		s = char(s,[col' repmat('-.',4,1)]);
		s = char(s,[col' repmat(':',4,1)]);
		s = char(s,s,s,s);
	end

	% When one contour should be plotted, two entries have to be given in
	% the contour plot (Matlab bug/feature).

	%if (~isempty(n)) & (length(n) == 1), n = [n n]; end
	
	% Setup the mesh-grid, use the axis of the currently active figure.
	% Note: this will be a 0-1 grid in case of no given scatterplot.
	
	hold on; V = axis; 
	gs = gridsize; dx = (V(2)-V(1))/gs; dy = (V(4)-V(3))/gs;
    if (plottype == 1)
			m = (gs+1); X = (V(1):dx:V(2))';
			D = double([X,zeros(m,k-1)]*w);
    else
			m = (gs+1)*(gs+1); [X Y] = meshgrid(V(1):dx:V(2),V(3):dy:V(4));
	    D = double([X(:),Y(:),zeros(m,k-2)]*w);
    end;

    if (~cnd)
        D = sum(D,2);
    end;
    
	% HH will contain all handles to graphics created in this routine.

	hh = [];

    % Plot the densities in case of 1D output.
    if (plottype == 1)
        for j = 1:size(D,2)
            if (size(s,1) > 1), ss = s(j,:); else ss = s; end

            % Plot the densities and add the handles to HH.

    		h = plot([V(1):dx:V(2)],D(:,j),deblank(ss));
    		hh = [hh; h];
        end    
        axis ([V(1) V(2) 0 1.2*max(max(D))]);
				ylabel('Density')
    end
    
	% Plot the contours in case of 2D output.
	if (plottype == 2) | (plottype == 4) | (plottype == 6)

		% Define the contour-heights if they are not given.

		if (isempty(n))
			n = 10;
			dmax = max(D(:)); dmin = min(D(:)); dd = (dmax-dmin)/(n+1);
			n = [dmin+dd:dd:dmax-dd];
		end;			

		if length(n) == 1, n = [n n]; end

		% Plot the contours for each of the classes.
	
   		for j = 1:size(D,2)

   			if (size(s,1) > 1), ss = s(j,:); else, ss = s; end

   			Z = reshape(D(:,j),gs+1,gs+1);

			% Plot the contours and add the handles to HH.

   			[cc, h] = contour([V(1):dx:V(2)],[V(3):dy:V(4)],Z,n,deblank(ss));
   			hh = [hh; h];
			
        end
        
		view(2);

	end

	% Plot the surface in case of 3D output.

	if (plottype == 3) | (plottype == 4) | (plottype == 5) | (plottype == 6)

		% Scale the outputs to cover the whole colour range.

		%E = D - min(D(:));
		%E = 255*E/max(E(:))+1;
        E = D; % Scaling appears disputable (RD)
		if (c>1)
        Z = reshape(sum(E,2),gs+1,gs+1);
	  else
        Z = reshape(E(:,1),gs+1,gs+1);
	  end 

		if (plottype == 4) | (plottype == 6)	
			Z = Z + max(max(Z));
		end;

		% Plot the surface, set up lighting and add the handles to HH.

		h = surf([V(1):dx:V(2)],[V(3):dy:V(4)],Z);

		if (plottype == 3) | (plottype == 4)
            colormap jet;
            shading interp;
			set(h,'FaceColor','interp','EdgeColor','none','FaceLighting','none ');
		else
			colormap white;
			shading faceted;
		end

		view(-37.5,20);
		camlight left; 								% Necessary to solve camlight bug?
		camlight headlight;
		camlight right;
		hh = [hh; h];

	end

	hold off; if (nargout > 0), handle = hh; end

return

